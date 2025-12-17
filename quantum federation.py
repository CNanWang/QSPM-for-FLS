from typing import Tuple, Dict, List
import random, base64, hashlib, io
import numpy as np
from cryptography.fernet import Fernet

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# helpers & stable sigmoid
def stable_sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    out = np.empty_like(x, dtype=float)
    pos_mask = x >= 0
    neg_mask = ~pos_mask
    out[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
    exp_x = np.exp(x[neg_mask])
    out[neg_mask] = exp_x / (1.0 + exp_x)
    return out

def serialize_array(arr: np.ndarray) -> bytes:
    bio = io.BytesIO()
    np.save(bio, arr, allow_pickle=False)
    bio.seek(0)
    return bio.read()

def deserialize_array(b: bytes) -> np.ndarray:
    bio = io.BytesIO(b)
    bio.seek(0)
    return np.load(bio, allow_pickle=False)

def derive_fernet_key_from_bits(bits: List[int]) -> bytes:
    bitstring = ''.join(str(b) for b in bits).encode('utf-8')
    digest = hashlib.sha256(bitstring).digest()
    return base64.urlsafe_b64encode(digest)

# Qiskit GHZ helpers 
def create_3qubit_ghz_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    return qc

def measure_ghz_in_basis_once(basis: str) -> Tuple[int, int, int]:
    qc = create_3qubit_ghz_circuit()
    if basis.upper() == 'X':
        qc.h(0); qc.h(1); qc.h(2)
    qc.measure([0,1,2], [0,1,2])
    sim = AerSimulator()
    job = sim.run(qc, shots=1)
    result = job.result()
    counts = result.get_counts()
    outcome = next(iter(counts.keys()))
    bits = tuple(int(ch) for ch in outcome[::-1])
    return bits

def measure_ghz_with_client_pauli_once(apply_x_on_B: bool, apply_x_on_C: bool) -> Tuple[int,int,int]:
    qc = create_3qubit_ghz_circuit()
    if apply_x_on_B:
        qc.x(1)
    if apply_x_on_C:
        qc.x(2)
    qc.measure([0,1,2], [0,1,2])
    sim = AerSimulator()
    job = sim.run(qc, shots=1)
    result = job.result()
    counts = result.get_counts()
    outcome = next(iter(counts.keys()))
    bits = tuple(int(ch) for ch in outcome[::-1])
    return bits

# Attack helper 
def intercept_resend_on_BC(triple: Tuple[int,int,int], p: float) -> Tuple[int,int,int]:
    a,b,c = triple
    if random.random() < p:
        b = random.randint(0,1)
    if random.random() < p:
        c = random.randint(0,1)
    return (a,b,c)

# logistic loss & grad 
def logistic_loss_and_grad(w: np.ndarray, X: np.ndarray, y: np.ndarray):
    z = X.dot(w)
    p = stable_sigmoid(z)
    eps = 1e-9
    loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    grad = X.T.dot(p - y)   # sum over local samples 
    return loss, grad, X.shape[0]

# Core one-mode runner 
def run_fedfl_one_mode(
    mode: str,   # 'quantum' or 'classical'
    X_train_all: np.ndarray,
    y_train_all: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_clients: int = 5,
    n_features: int = 12,
    fl_rounds: int = 12,
    lr: float = 0.01,
    random_seed: int = 2025,
    ghz_d: int = 4,
    client_private_key_len: int = 64,
    ghz_threshold: float = 0.10,
    attack_model: str = 'none',
    p_attack: float = 0.0,
    max_retries: int = 3,
    verbose: bool = True,
    # Adam / reg params
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_eps: float = 1e-8,
    weight_decay: float = 1e-4
):
    assert mode in ('quantum', 'classical')
    np.random.seed(random_seed); random.seed(random_seed)

    # split dataset among clients
    clients_data = []
    per = len(X_train_all)//n_clients
    start = 0
    for i in range(n_clients):
        end = start + per
        if i == n_clients - 1:
            end = len(X_train_all)
        clients_data.append((X_train_all[start:end], y_train_all[start:end]))
        start = end

    # server weights and Adam moment estimates
    server_w = np.zeros(n_features)
    m_t = np.zeros_like(server_w)
    v_t = np.zeros_like(server_w)
    adam_t = 0

    client_bits_map: Dict[int, List[int]] = {}
    client_fernets: Dict[int, Fernet] = {}
    server_recovered_keys: Dict[int, List[int]] = {}

    #  quantum key establishment 
    def quantum_key_establish_for_client_with_retry(cid: int, seed_offset: int = 0) -> List[int]:
        attempt = 0
        L = client_private_key_len
        n = L // 2
        while attempt < max_retries:
            attempt += 1
            if verbose:
                print(f"[Q-TRY] client{cid} attempt {attempt}/{max_retries} ...")
            m = n + ghz_d
            # security check
            checked_indices = random.sample(range(m), ghz_d)
            errors = 0; attacked_checks = 0
            for idx in checked_indices:
                basis = random.choice(['X','Z'])
                a,b,c = measure_ghz_in_basis_once(basis)
                if attack_model == 'intercept_resend' and p_attack > 0:
                    a2,b2,c2 = intercept_resend_on_BC((a,b,c), p_attack)
                else:
                    a2,b2,c2 = (a,b,c)
                passed = (a2 == b2 == c2) if basis == 'Z' else ((a2 ^ b2 ^ c2) == 0)
                if not passed:
                    errors += 1
                if (a2,b2,c2) != (a,b,c):
                    attacked_checks += 1
                if verbose:
                    print(f"[Q-CHK] idx{idx} basis={basis} ideal={(a,b,c)} recv={(a2,b2,c2)} pass={passed}")
            error_rate = errors / float(ghz_d)
            if verbose:
                print(f"[Q] attempt {attempt} errors={errors}/{ghz_d} err_rate={error_rate:.4f} attacked_checks={attacked_checks}/{ghz_d}")
            if error_rate >= ghz_threshold:
                if attempt < max_retries:
                    if verbose: print("[Q-RETRY] security check failed -> retry")
                    continue
                else:
                    raise RuntimeError(f"Quantum channel insecure after {max_retries} attempts for client {cid}")
            kept_indices = [i for i in range(m) if i not in checked_indices][:n]
            client_bits = [int(x) for x in np.random.randint(0,2,size=2*len(kept_indices)).tolist()]
            if verbose:
                print(f"[Q-KEYGEN] client{cid} client_bits len={len(client_bits)} first12={client_bits[:12]}")
            received_triples = {}
            tampered = False; attacked_pairs = 0
            for i, idx in enumerate(kept_indices):
                bit_for_B = client_bits[2*i]; bit_for_C = client_bits[2*i+1]
                a_meas, b_meas, c_meas = measure_ghz_with_client_pauli_once(apply_x_on_B=(bit_for_B==1), apply_x_on_C=(bit_for_C==1))
                if attack_model == 'intercept_resend' and p_attack > 0:
                    a2,b2,c2 = intercept_resend_on_BC((a_meas,b_meas,c_meas), p_attack)
                else:
                    a2,b2,c2 = (a_meas,b_meas,c_meas)
                if verbose and i < 6:
                    print(f"[Q-KEY] client{cid} idx{i} bits=(B:{bit_for_B},C:{bit_for_C}) measured=({a_meas}, {b_meas}, {c_meas}) after_attack=({a2}, {b2}, {c2})")
                if (a_meas, b_meas, c_meas) != (a2, b2, c2):
                    tampered = True
                    if verbose:
                        print(f"[Q-TAMPER] client{cid} idx{i} measured={(a_meas,b_meas,c_meas)} after_attack={(a2,b2,c2)} -> abort")
                    attacked_pairs += 1
                    break
                received_triples[idx] = (a2,b2,c2)
            if tampered:
                if attempt < max_retries:
                    if verbose:
                        print(f"[Q-ABORT] Tampering detected for client{cid} on attempt {attempt}. Will retry.")
                    continue
                else:
                    raise RuntimeError(f"Quantum channel tampered repeatedly for client {cid}")
            recovered_bits = []
            for idx in kept_indices:
                recA, recB, recC = received_triples[idx]
                recovered_bits.append(1 if recB != recA else 0)
                recovered_bits.append(1 if recC != recA else 0)
            if verbose:
                client_true_bits_str = ''.join(str(b) for b in client_bits)
                recovered_bits_str = ''.join(str(b) for b in recovered_bits)
                recovered_mismatch = sum(1 for a,b in zip(recovered_bits, client_bits) if a != b)
                print(f"\n[Q-RESULT] Client {cid} attempt {attempt} key-transfer finished (no tamper).")
                print(f"  DEBUG client_true_bits (len={len(client_bits)}): {client_true_bits_str}")
                print(f"  SERVER recovered_bits  (len={len(recovered_bits)}): {recovered_bits_str}")
                print(f"  recovered_mismatch = {recovered_mismatch} / {len(client_bits)}")
                print(f"  attacked_pairs (during transfer) = {attacked_pairs} / {len(kept_indices)}\n")
            server_recovered_keys[cid] = recovered_bits
            return recovered_bits
        raise RuntimeError("Unexpected quantum establishment flow")

    # classical key establishment (ideal)
    def classical_key_establish_for_client(cid: int) -> List[int]:
        L = client_private_key_len
        client_bits = [int(x) for x in np.random.randint(0,2,size=L).tolist()]
        if verbose:
            print(f"[C-INIT] client{cid} classical key len={L}")
        server_recovered_keys[cid] = client_bits.copy()
        return client_bits

    # classical XOR update (with logs)
    def classical_xor_update_for_client(cid: int) -> Tuple[List[int], List[int]]:
        prev_bits = client_bits_map[cid]
        L = len(prev_bits)
        client_new_bits = [int(x) for x in np.random.randint(0,2,size=L).tolist()]
        payload = [nb ^ pb for nb,pb in zip(client_new_bits, prev_bits)]
        recovered_new = [pb ^ pl for pb,pl in zip(prev_bits, payload)]
        if verbose:
            mismatches = sum(1 for a,b in zip(client_new_bits, recovered_new) if a!=b)
            print(f"[C] Client {cid} XOR update:")
            print(f"  DEBUG client_new_bits (len={L}) first12={client_new_bits[:12]}")
            print(f"  SERVER recovered_new  (len={L}) first12={recovered_new[:12]} mismatches={mismatches}/{L}")
        return client_new_bits, recovered_new

    # initial key establishment (mode-dependent)
    for cid in range(len(clients_data)):
        if mode == 'quantum':
            recovered_bits = quantum_key_establish_for_client_with_retry(cid, seed_offset=0)
        else:
            recovered_bits = classical_key_establish_for_client(cid)
        client_bits_map[cid] = recovered_bits
        fkey = derive_fernet_key_from_bits(recovered_bits)
        client_fernets[cid] = Fernet(fkey)

    # client wrapper
    class ClientLocal:
        def __init__(self, cid:int, Xloc:np.ndarray, yloc:np.ndarray, fernet: Fernet):
            self.id = cid
            self.X = Xloc; self.y = yloc; self.n = Xloc.shape[0]; self.fernet = fernet
        def compute_local_grad(self, w: np.ndarray):
            loss, grad_sum, n = logistic_loss_and_grad(w, self.X, self.y)
            return grad_sum, loss, n
        def encrypt_arr(self, arr: np.ndarray) -> bytes:
            return self.fernet.encrypt(serialize_array(arr))
        def decrypt_arr(self, token: bytes) -> np.ndarray:
            return deserialize_array(self.fernet.decrypt(token))

    clients = [ClientLocal(cid, clients_data[cid][0], clients_data[cid][1], client_fernets[cid]) for cid in range(len(clients_data))]
    server_fernets = {cid: client_fernets[cid] for cid in client_fernets}

    rng = np.random.default_rng(random_seed + 9999)
    rounds = []; train_losses = []; test_losses = []; test_accs = []

    # FL loop with Adam update on server
    for rnd in range(1, fl_rounds + 1):
        if verbose:
            print(f"\n--- {mode.upper()} FL Round {rnd}/{fl_rounds} ---")
        # key schedule
        if (rnd - 1) % 3 == 0 and rnd != 1:
            if verbose: print("[SCHEDULE] re-key (full channel)")
            for cid in range(len(clients_data)):
                if mode == 'quantum':
                    recovered_bits = quantum_key_establish_for_client_with_retry(cid, seed_offset=rnd)
                else:
                    recovered_bits = classical_key_establish_for_client(cid)
                client_bits_map[cid] = recovered_bits
                server_recovered_keys[cid] = recovered_bits.copy()
                fkey = derive_fernet_key_from_bits(recovered_bits)
                client_fernets[cid] = Fernet(fkey)
                clients[cid].fernet = client_fernets[cid]
                server_fernets[cid] = client_fernets[cid]
        elif (rnd - 1) % 3 in (1,2):
            if verbose: print("[SCHEDULE] Classic XOR update")
            for cid in range(len(clients_data)):
                client_new_bits, recovered_new = classical_xor_update_for_client(cid)
                client_bits_map[cid] = recovered_new
                server_recovered_keys[cid] = recovered_new.copy()
                fkey = derive_fernet_key_from_bits(recovered_new)
                client_fernets[cid] = Fernet(fkey)
                clients[cid].fernet = client_fernets[cid]
                server_fernets[cid] = client_fernets[cid]

        # compute pairwise masks
        pairwise_masks: Dict[int, Dict[int, np.ndarray]] = {}
        for c in clients:
            other_ids = [o.id for o in clients if o.id != c.id]
            masks = {j: rng.normal(loc=0.0, scale=1.0, size=n_features) for j in other_ids}
            pairwise_masks[c.id] = masks

        # clients compute local masked grads and encrypt
        encrypted_messages: Dict[int, bytes] = {}
        total_samples = 0
        for c in clients:
            grad_sum, loss, ns = c.compute_local_grad(server_w)
            total_samples += ns
            mask_out = np.sum(np.stack(list(pairwise_masks[c.id].values())), axis=0) if pairwise_masks[c.id] else np.zeros(n_features)
            mask_in = np.zeros(n_features)
            for j in pairwise_masks:
                if j == c.id: continue
                mask_in += pairwise_masks[j][c.id]
            final_mask = mask_out - mask_in
            masked_grad = grad_sum + final_mask
            token = c.encrypt_arr(masked_grad)
            encrypted_messages[c.id] = token

        # server decrypt and aggregate
        aggregated = np.zeros(n_features)
        for cid, token in encrypted_messages.items():
            f = server_fernets[cid]
            arr = deserialize_array(f.decrypt(token))
            aggregated += arr

        # average gradient across total samples
        avg_grad = aggregated / float(total_samples)

        # add L2 regularization gradient: grad += weight_decay * w
        if weight_decay > 0:
            avg_grad = avg_grad + weight_decay * server_w

        # Adam update
        adam_t += 1
        m_t = adam_beta1 * m_t + (1 - adam_beta1) * avg_grad
        v_t = adam_beta2 * v_t + (1 - adam_beta2) * (avg_grad * avg_grad)
        m_hat = m_t / (1 - adam_beta1 ** adam_t)
        v_hat = v_t / (1 - adam_beta2 ** adam_t)
        server_w = server_w - lr * (m_hat / (np.sqrt(v_hat) + adam_eps))

        # server sends updated model to clients (verify)
        for c in clients:
            f = server_fernets[c.id]
            token = f.encrypt(serialize_array(server_w))
            _ = c.decrypt_arr(token)

        # evaluation
        train_loss, _, _ = logistic_loss_and_grad(server_w, X_train_all, y_train_all)
        z_test = X_test.dot(server_w)
        p_test = stable_sigmoid(z_test)
        eps = 1e-9
        test_loss = -np.mean(y_test * np.log(p_test + eps) + (1 - y_test) * np.log(1 - p_test + eps))
        preds = (p_test >= 0.5).astype(int)
        test_acc = float(np.mean(preds == y_test))

        rounds.append(rnd); train_losses.append(train_loss); test_losses.append(test_loss); test_accs.append(test_acc)
        if verbose:
            print(f"[{mode.upper()} FL] Round {rnd}: train_loss={train_loss:.4f} test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

    history = {
        'rounds': rounds,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'server_w': server_w,
        'client_bits_map': client_bits_map,
        'server_recovered_keys': server_recovered_keys
    }
    return history

# Top-level comparer 
def compare_quantum_vs_classical(
    n_clients: int = 5,
    samples_per_client: int = 300,   
    n_features: int = 12,
    fl_rounds: int = 12,
    lr: float = 0.01,
    random_seed: int = 2025,
    ghz_d: int = 4,
    client_private_key_len: int = 64,
    ghz_threshold: float = 0.10,
    attack_model: str = 'none',
    p_attack: float = 0.0,
    max_retries: int = 3,
    verbose: bool = True
):
    X, y = make_classification(n_samples=n_clients * samples_per_client,
                               n_features=n_features,
                               n_informative=max(2, int(n_features*0.8)),
                               n_redundant=0,
                               n_clusters_per_class=1,
                               class_sep=1.5,
                               random_state=random_seed)
    X_train_all, X_test, y_train_all, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)
    scaler = StandardScaler().fit(X_train_all)
    X_train_all = scaler.transform(X_train_all)
    X_test = scaler.transform(X_test)

    history_q = run_fedfl_one_mode(
        mode='quantum',
        X_train_all=X_train_all, y_train_all=y_train_all, X_test=X_test, y_test=y_test,
        n_clients=n_clients, n_features=n_features, fl_rounds=fl_rounds, lr=lr,
        random_seed=random_seed, ghz_d=ghz_d, client_private_key_len=client_private_key_len,
        ghz_threshold=ghz_threshold, attack_model=attack_model, p_attack=p_attack, max_retries=max_retries,
        verbose=verbose
    )

    history_c = run_fedfl_one_mode(
        mode='classical',
        X_train_all=X_train_all, y_train_all=y_train_all, X_test=X_test, y_test=y_test,
        n_clients=n_clients, n_features=n_features, fl_rounds=fl_rounds, lr=lr,
        random_seed=random_seed, ghz_d=ghz_d, client_private_key_len=client_private_key_len,
        ghz_threshold=ghz_threshold, attack_model='none', p_attack=0.0, max_retries=max_retries,
        verbose=verbose
    )

    rounds = history_q['rounds']
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(rounds, history_q['train_losses'],  color='#1f77b4', linewidth=2, label='quantum train_loss')
    plt.plot(rounds, history_q['test_losses'],  color='#4c8cff', linewidth=2, label='quantum test_loss')
    plt.plot(rounds, history_c['train_losses'], '--',color='#d62728', linewidth=2, label='classical train_loss')
    plt.plot(rounds, history_c['test_losses'], '--', color='#ff6b6b', linewidth=2,label='classical test_loss')
    plt.xlabel('round'); plt.ylabel('loss'); plt.title('Loss: quantum vs classical'); plt.legend()
    plt.grid(True, alpha=0.8, linestyle='-', linewidth=0.5) 
    plt.subplot(1,2,2)
    plt.plot(rounds, history_q['test_accs'], color='#1f77b4', linewidth=2, label='quantum test_acc')
    plt.plot(rounds, history_c['test_accs'], '--', color='#d62728', linewidth=2, label='classical test_acc')
    plt.xlabel('round'); plt.ylabel('accuracy'); plt.ylim(0,1.0); plt.title('Test Accuracy: quantum vs classical'); plt.legend()
    plt.grid(True, alpha=0.8, linestyle='-', linewidth=0.5) 
    plt.tight_layout(); plt.show()
 

    return history_q, history_c

# main 
if __name__ == "__main__":
    hist_q, hist_c = compare_quantum_vs_classical(
        n_clients=5,
        samples_per_client=300,
        n_features=12,
        fl_rounds=9,
        lr=0.01,
        random_seed=2025,
        ghz_d=10,
        client_private_key_len=64,
        ghz_threshold=0.01,
        attack_model='intercept_resend',# 'none' or 'intercept_resend'
        p_attack=0.01,
        max_retries=5,
        verbose=True
    )

    print("\n=== Final comparison (last round) ===")
    print(f"Quantum: test_loss={hist_q['test_losses'][-1]:.4f} train_loss={hist_q['train_losses'][-1]:.4f} test_acc={hist_q['test_accs'][-1]:.4f} ")
    print(f"Classical: test_loss={hist_c['test_losses'][-1]:.4f} train_loss={hist_c['train_losses'][-1]:.4f} test_acc={hist_c['test_accs'][-1]:.4f} ")
