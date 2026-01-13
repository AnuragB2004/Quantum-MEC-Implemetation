"""
Quantum Game Theory for Edge Resource Allocation
Complete Implementation for Kaggle GPU T4 x2

Requirements:
- Python 3.8+
- numpy, scipy, matplotlib, qiskit
- pandas for data logging

Run on Kaggle with GPU T4 x2 for accelerated simulation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import time
from collections import defaultdict
import json

# Quantum simulation (using numpy for NISQ simulation)
class QuantumCircuit:
    """Simplified quantum circuit simulator for VQC"""
    
    def __init__(self, n_qubits: int, n_layers: int = 2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.state = None
        
    def initialize(self, gamma: float, topology: str = 'ring'):
        """Initialize entangled state"""
        # Start with |0...0⟩
        self.state = np.zeros(2**self.n_qubits, dtype=complex)
        self.state[0] = 1.0
        
        # Apply entangling operator J(gamma)
        self.apply_entangling_gate(gamma, topology)
        
    def apply_entangling_gate(self, gamma: float, topology: str):
        """Apply J(gamma) = exp(i*gamma * sum ZZ)"""
        n = self.n_qubits
        dim = 2**n
        
        # Get pairs based on topology
        if topology == 'ring':
            pairs = [(i, (i+1) % n) for i in range(n)]
        elif topology == 'star':
            pairs = [(0, i) for i in range(1, n)]
        elif topology == 'all-to-all':
            pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
        else:
            pairs = [(i, (i+1) % n) for i in range(n)]
        
        # Build ZZ operator
        H = np.zeros((dim, dim), dtype=complex)
        for i, j in pairs:
            # ZZ interaction
            for state_idx in range(dim):
                bit_i = (state_idx >> (n-1-i)) & 1
                bit_j = (state_idx >> (n-1-j)) & 1
                parity = (-1)**(bit_i + bit_j)
                H[state_idx, state_idx] += parity / 2
        
        # Apply exp(i*gamma*H)
        U = np.eye(dim, dtype=complex)
        for k in range(1, 10):  # Taylor expansion
            U += (1j * gamma * H)**k / np.math.factorial(k)
        
        self.state = U @ self.state
        
    def apply_local_unitaries(self, params: np.ndarray):
        """Apply local U_i(theta_i) to each qubit"""
        n = self.n_qubits
        dim = 2**n
        
        for i in range(n):
            theta, phi, psi = params[3*i:3*i+3]
            
            # Single qubit unitary: Rz(psi) Ry(theta) Rz(phi)
            U_single = self._rz(psi) @ self._ry(theta) @ self._rz(phi)
            
            # Extend to full space
            U_full = self._extend_unitary(U_single, i, n)
            self.state = U_full @ self.state
    
    def _ry(self, theta: float) -> np.ndarray:
        """Rotation around Y axis"""
        c, s = np.cos(theta/2), np.sin(theta/2)
        return np.array([[c, -s], [s, c]], dtype=complex)
    
    def _rz(self, phi: float) -> np.ndarray:
        """Rotation around Z axis"""
        return np.array([[np.exp(-1j*phi/2), 0], 
                        [0, np.exp(1j*phi/2)]], dtype=complex)
    
    def _extend_unitary(self, U: np.ndarray, qubit_idx: int, n_qubits: int) -> np.ndarray:
        """Extend single-qubit unitary to full Hilbert space"""
        dim = 2**n_qubits
        U_full = np.eye(dim, dtype=complex)
        
        for state_idx in range(dim):
            bit = (state_idx >> (n_qubits-1-qubit_idx)) & 1
            # Apply U based on this qubit's state
            for new_bit in range(2):
                new_state = state_idx ^ ((bit ^ new_bit) << (n_qubits-1-qubit_idx))
                U_full[new_state, state_idx] = U[new_bit, bit]
        
        return U_full
    
    def measure(self, n_shots: int = 1000) -> Dict[int, int]:
        """Measure all qubits in computational basis"""
        probs = np.abs(self.state)**2
        probs = probs / np.sum(probs)  # Normalize
        
        outcomes = np.random.choice(
            len(probs), 
            size=n_shots, 
            p=probs
        )
        
        counts = {}
        for outcome in outcomes:
            counts[outcome] = counts.get(outcome, 0) + 1
        
        return counts
    
    def get_probability_distribution(self) -> np.ndarray:
        """Get exact probability distribution"""
        return np.abs(self.state)**2


@dataclass
class Task:
    """Computation task"""
    cpu_cycles: float  # CPU cycles required
    data_size: float   # Input data size in bytes
    deadline: float    # Latency deadline in seconds
    device_id: int


@dataclass
class DeviceConfig:
    """Mobile device configuration"""
    local_cpu_freq: float  # Local CPU frequency (Hz)
    tx_power: float        # Transmission power (W)
    alpha: float           # Latency-energy weight
    beta: float            # Deadline violation penalty


@dataclass
class EdgeConfig:
    """Edge node configuration"""
    cpu_capacity: float    # CPU capacity (Hz)
    max_queue_size: int


class MECSystem:
    """Mobile Edge Computing system simulator"""
    
    def __init__(
        self,
        n_devices: int = 8,
        n_edges: int = 2,
        bandwidth: float = 20e6,  # 20 MHz
        noise_psd: float = 1e-9,  # W/Hz
        kappa: float = 1e-28,     # Capacitance coefficient
    ):
        self.n_devices = n_devices
        self.n_edges = n_edges
        self.bandwidth = bandwidth
        self.noise_psd = noise_psd
        self.kappa = kappa
        
        # Device configurations
        self.devices = [
            DeviceConfig(
                local_cpu_freq=1e9,  # 1 GHz
                tx_power=0.5,        # 0.5 W
                alpha=0.7,           # Latency priority
                beta=100.0           # High penalty
            ) for _ in range(n_devices)
        ]
        
        # Edge configurations
        self.edges = [
            EdgeConfig(
                cpu_capacity=10e9,   # 10 GHz
                max_queue_size=20
            ) for _ in range(n_edges)
        ]
        
        # Channel gains (Rayleigh fading)
        self.channel_gains = np.random.rayleigh(1.0, (n_devices, n_edges))
        
        # Edge loads
        self.edge_loads = np.zeros(n_edges)
        
    def generate_tasks(self, 
                       c_min: float = 1e8, 
                       c_max: float = 5e8,
                       b_min: float = 1e5,
                       b_max: float = 1e6) -> List[Task]:
        """Generate random tasks for all devices"""
        tasks = []
        for i in range(self.n_devices):
            task = Task(
                cpu_cycles=np.random.uniform(c_min, c_max),
                data_size=np.random.uniform(b_min, b_max),
                deadline=0.1,  # 100ms deadline
                device_id=i
            )
            tasks.append(task)
        return tasks
    
    def compute_transmission_delay(self, device_id: int, edge_id: int, 
                                   data_size: float) -> float:
        """Compute wireless transmission delay"""
        P_i = self.devices[device_id].tx_power
        h = self.channel_gains[device_id, edge_id]
        
        # Shannon capacity
        rate = self.bandwidth * np.log2(1 + P_i * h / (self.noise_psd * self.bandwidth))
        
        return data_size / rate
    
    def compute_queueing_delay(self, edge_id: int) -> float:
        """Compute queueing delay (M/M/1 approximation)"""
        load = self.edge_loads[edge_id]
        capacity = self.edges[edge_id].cpu_capacity
        avg_cycles = 3e8  # Average task size
        
        arrival_rate = load
        service_rate = capacity / avg_cycles
        
        if arrival_rate >= service_rate * 0.9:  # Near saturation
            return 0.5  # High delay
        
        return arrival_rate / (service_rate - arrival_rate) if arrival_rate < service_rate else 1.0
    
    def compute_service_delay(self, task: Task, edge_id: int) -> float:
        """Compute service delay on edge"""
        # Fair share of CPU
        n_tasks = max(1, self.edge_loads[edge_id])
        cpu_share = self.edges[edge_id].cpu_capacity / n_tasks
        
        return task.cpu_cycles / cpu_share
    
    def compute_local_delay(self, task: Task, device_id: int) -> float:
        """Compute local execution delay"""
        freq = self.devices[device_id].local_cpu_freq
        return task.cpu_cycles / freq
    
    def compute_local_energy(self, task: Task, device_id: int) -> float:
        """Compute local execution energy"""
        freq = self.devices[device_id].local_cpu_freq
        delay = self.compute_local_delay(task, device_id)
        return self.kappa * (freq ** 3) * delay
    
    def compute_offload_energy(self, task: Task, device_id: int, edge_id: int) -> float:
        """Compute offloading energy (transmission only)"""
        tx_delay = self.compute_transmission_delay(device_id, edge_id, task.data_size)
        return self.devices[device_id].tx_power * tx_delay
    
    def compute_utility(self, tasks: List[Task], actions: np.ndarray) -> np.ndarray:
        """Compute utilities for all devices given actions"""
        utilities = np.zeros(self.n_devices)
        
        # Update edge loads
        self.edge_loads = np.zeros(self.n_edges)
        for device_id, action in enumerate(actions):
            if action > 0:  # Offloaded
                edge_id = action - 1
                self.edge_loads[edge_id] += 1
        
        for device_id, (task, action) in enumerate(zip(tasks, actions)):
            if action == 0:  # Local execution
                latency = self.compute_local_delay(task, device_id)
                energy = self.compute_local_energy(task, device_id)
            else:  # Offload to edge
                edge_id = action - 1
                tx_delay = self.compute_transmission_delay(device_id, edge_id, task.data_size)
                queue_delay = self.compute_queueing_delay(edge_id)
                svc_delay = self.compute_service_delay(task, edge_id)
                latency = tx_delay + queue_delay + svc_delay
                energy = self.compute_offload_energy(task, device_id, edge_id)
            
            # Normalize energy
            E_max = self.compute_local_energy(task, device_id)
            energy_norm = energy / max(E_max, 1e-6)
            
            # Deadline violation penalty
            violation = 1.0 if latency > task.deadline else 0.0
            
            # Utility (negative cost)
            alpha = self.devices[device_id].alpha
            beta = self.devices[device_id].beta
            utilities[device_id] = -(alpha * latency + (1-alpha) * energy_norm + beta * violation)
        
        return utilities
    
    def compute_imbalance(self) -> float:
        """Compute load imbalance metric"""
        if np.mean(self.edge_loads) < 0.1:
            return 0.0
        return (np.max(self.edge_loads) - np.min(self.edge_loads)) / np.mean(self.edge_loads)


class QuantumNashSolver:
    """Quantum Nash Equilibrium solver with VQC"""
    
    def __init__(
        self,
        n_devices: int,
        n_layers: int = 2,
        topology: str = 'ring',
        gamma_min: float = 0.0,
        gamma_max: float = np.pi/2,
        tau_0: float = 0.1,
        max_iters: int = 50,
        epsilon: float = 0.05,
        n_shots: int = 200,
        seed: int = 42
    ):
        self.n_devices = n_devices
        self.n_layers = n_layers
        self.topology = topology
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.tau_0 = tau_0
        self.max_iters = max_iters
        self.epsilon = epsilon
        self.n_shots = n_shots
        
        np.random.seed(seed)
        
        # Parameters: 3 angles per device
        self.n_params = 3 * n_devices
        
    def adaptive_gamma(self, imbalance: float, eta: float = 2.0) -> float:
        """Compute adaptive entanglement level"""
        sigmoid = 1 / (1 + np.exp(-eta * imbalance))
        return self.gamma_min + (self.gamma_max - self.gamma_min) * sigmoid
    
    def run_vqc(self, params: np.ndarray, gamma: float) -> np.ndarray:
        """Execute VQC and return probability distribution"""
        qc = QuantumCircuit(self.n_devices, self.n_layers)
        qc.initialize(gamma, self.topology)
        qc.apply_local_unitaries(params)
        
        # Get exact probabilities for efficiency
        probs = qc.get_probability_distribution()
        
        # Sample for noise simulation
        if self.n_shots < np.inf:
            counts = qc.measure(self.n_shots)
            sampled_probs = np.zeros(2**self.n_devices)
            for state, count in counts.items():
                sampled_probs[state] = count / self.n_shots
            return sampled_probs
        
        return probs
    
    def decode_actions(self, state_idx: int) -> np.ndarray:
        """Decode state index to action vector"""
        # For binary offloading: bit=0 -> local, bit=1 -> edge 0
        actions = np.zeros(self.n_devices, dtype=int)
        for i in range(self.n_devices):
            bit = (state_idx >> (self.n_devices - 1 - i)) & 1
            actions[i] = bit  # 0=local, 1=edge_0 (for 2 edges, map to edge 0 or 1 alternately)
        return actions
    
    def compute_expected_utility(
        self, 
        prob_dist: np.ndarray, 
        mec_system: MECSystem,
        tasks: List[Task]
    ) -> np.ndarray:
        """Compute expected utility for each device"""
        expected_utils = np.zeros(self.n_devices)
        
        for state_idx, prob in enumerate(prob_dist):
            if prob < 1e-8:
                continue
            
            actions = self.decode_actions(state_idx)
            # Map binary actions to edges (simple mapping)
            actions_mapped = actions + (actions > 0) * (state_idx % 2)  # Distribute between edges
            
            utils = mec_system.compute_utility(tasks, actions_mapped)
            expected_utils += prob * utils
        
        return expected_utils
    
    def estimate_regret(
        self,
        params: np.ndarray,
        gamma: float,
        mec_system: MECSystem,
        tasks: List[Task],
        n_deviations: int = 5
    ) -> float:
        """Estimate total regret by sampling deviations"""
        prob_dist = self.run_vqc(params, gamma)
        base_utils = self.compute_expected_utility(prob_dist, mec_system, tasks)
        
        total_regret = 0.0
        
        for i in range(self.n_devices):
            max_gain = 0.0
            
            for _ in range(n_deviations):
                # Sample random deviation for device i
                dev_params = params.copy()
                dev_params[3*i:3*i+3] += np.random.randn(3) * 0.5
                dev_params[3*i:3*i+3] = np.clip(dev_params[3*i:3*i+3], 0, 2*np.pi)
                
                dev_prob_dist = self.run_vqc(dev_params, gamma)
                dev_utils = self.compute_expected_utility(dev_prob_dist, mec_system, tasks)
                
                gain = dev_utils[i] - base_utils[i]
                max_gain = max(max_gain, gain)
            
            total_regret += max(0, max_gain)
        
        return total_regret
    
    def compute_entropy(self, prob_dist: np.ndarray) -> float:
        """Compute Shannon entropy of distribution"""
        prob_dist = prob_dist[prob_dist > 1e-10]
        return -np.sum(prob_dist * np.log(prob_dist + 1e-10))
    
    def solve(
        self,
        mec_system: MECSystem,
        tasks: List[Task],
        verbose: bool = True
    ) -> Tuple[np.ndarray, Dict]:
        """Solve for Quantum Nash Equilibrium"""
        
        # Adaptive gamma
        imbalance = mec_system.compute_imbalance()
        gamma = self.adaptive_gamma(imbalance)
        
        # Initialize parameters
        params = np.random.uniform(0, 2*np.pi, self.n_params)
        
        # Optimization history
        history = {
            'regret': [],
            'entropy': [],
            'loss': [],
            'utilities': []
        }
        
        tau = self.tau_0
        
        for iteration in range(self.max_iters):
            # Compute objective
            prob_dist = self.run_vqc(params, gamma)
            regret = self.estimate_regret(params, gamma, mec_system, tasks)
            entropy = self.compute_entropy(prob_dist)
            loss = regret - tau * entropy
            
            utils = self.compute_expected_utility(prob_dist, mec_system, tasks)
            
            history['regret'].append(regret)
            history['entropy'].append(entropy)
            history['loss'].append(loss)
            history['utilities'].append(utils.copy())
            
            if verbose and iteration % 10 == 0:
                print(f"Iter {iteration}: Regret={regret:.4f}, Entropy={entropy:.4f}, "
                      f"MeanUtil={np.mean(utils):.4f}")
            
            # Convergence check
            if regret < self.epsilon:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
            
            # Optimization step (simplified gradient-free)
            def objective(p):
                pd = self.run_vqc(p, gamma)
                r = self.estimate_regret(p, gamma, mec_system, tasks, n_deviations=3)
                e = self.compute_entropy(pd)
                return r - tau * e
            
            # Simple random search for efficiency
            best_loss = loss
            best_params = params.copy()
            
            for _ in range(10):  # 10 random neighbors
                candidate = params + np.random.randn(self.n_params) * 0.3
                candidate = np.clip(candidate, 0, 2*np.pi)
                cand_loss = objective(candidate)
                
                if cand_loss < best_loss:
                    best_loss = cand_loss
                    best_params = candidate
            
            params = best_params
            
            # Anneal temperature
            tau = self.tau_0 / (1 + iteration / 20)
        
        # Final high-fidelity evaluation
        final_prob_dist = self.run_vqc(params, gamma)
        
        return params, {
            'prob_dist': final_prob_dist,
            'gamma': gamma,
            'history': history,
            'converged': regret < self.epsilon,
            'iterations': iteration + 1
        }


class ClassicalBaseline:
    """Classical best-response dynamics"""
    
    def __init__(self, n_devices: int, max_iters: int = 100):
        self.n_devices = n_devices
        self.max_iters = max_iters
    
    def solve(
        self,
        mec_system: MECSystem,
        tasks: List[Task],
        verbose: bool = False
    ) -> Tuple[np.ndarray, Dict]:
        """Best-response dynamics"""
        
        # Initialize random actions
        actions = np.random.randint(0, mec_system.n_edges + 1, self.n_devices)
        
        history = {'utilities': []}
        
        for iteration in range(self.max_iters):
            improved = False
            
            # Each device best-responds
            for i in range(self.n_devices):
                current_util = mec_system.compute_utility(tasks, actions)[i]
                best_action = actions[i]
                best_util = current_util
                
                # Try all actions
                for action in range(mec_system.n_edges + 1):
                    test_actions = actions.copy()
                    test_actions[i] = action
                    test_util = mec_system.compute_utility(tasks, test_actions)[i]
                    
                    if test_util > best_util:
                        best_util = test_util
                        best_action = action
                
                if best_action != actions[i]:
                    actions[i] = best_action
                    improved = True
            
            utils = mec_system.compute_utility(tasks, actions)
            history['utilities'].append(utils.copy())
            
            if verbose and iteration % 20 == 0:
                print(f"Iter {iteration}: MeanUtil={np.mean(utils):.4f}")
            
            if not improved:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
        
        return actions, {
            'history': history,
            'iterations': iteration + 1
        }


def run_experiment(
    n_devices: int = 8,
    n_edges: int = 2,
    n_episodes: int = 20,
    method: str = 'quantum'
) -> Dict:
    """Run full experiment"""
    
    print(f"\n{'='*60}")
    print(f"Running {method.upper()} method with N={n_devices}, M={n_edges}")
    print(f"{'='*60}\n")
    
    # Initialize system
    mec_system = MECSystem(n_devices=n_devices, n_edges=n_edges)
    
    # Metrics
    latencies = []
    p95_latencies = []
    energies = []
    violations = []
    iterations_list = []
    times = []
    utilities_list = []
    
    for episode in range(n_episodes):
        print(f"\nEpisode {episode+1}/{n_episodes}")
        
        # Generate tasks
        tasks = mec_system.generate_tasks()
        
        # Reset channel gains
        mec_system.channel_gains = np.random.rayleigh(1.0, (n_devices, n_edges))
        
        start_time = time.time()
        
        if method == 'quantum':
            solver = QuantumNashSolver(
                n_devices=n_devices,
                n_layers=2,
                topology='ring',
                max_iters=50,
                epsilon=0.05,
                n_shots=200
            )
            params, results = solver.solve(mec_system, tasks, verbose=False)
            
            # Sample actions from distribution
            prob_dist = results['prob_dist']
            state_idx = np.random.choice(len(prob_dist), p=prob_dist)
            actions = solver.decode_actions(state_idx)
            actions = actions + (actions > 0) * (state_idx % n_edges)
            
            iters = results['iterations']
            
        else:  # classical
            solver = ClassicalBaseline(n_devices=n_devices, max_iters=100)
            actions, results = solver.solve(mec_system, tasks, verbose=False)
            iters = results['iterations']
        
        elapsed = time.time() - start_time
        
        # Compute metrics
        utils = mec_system.compute_utility(tasks, actions)
        utilities_list.append(np.mean(utils))
        
        # Compute latencies
        episode_latencies = []
        episode_energies = []
        episode_violations = 0
        
        for device_id, (task, action) in enumerate(zip(tasks, actions)):
            if action == 0:  # Local
                lat = mec_system.compute_local_delay(task, device_id)
                eng = mec_system.compute_local_energy(task, device_id)
            else:  # Offload
                edge_id = min(action - 1, n_edges - 1)
                tx_delay = mec_system.compute_transmission_delay(device_id, edge_id, task.data_size)
                queue_delay = mec_system.compute_queueing_delay(edge_id)
                svc_delay = mec_system.compute_service_delay(task, edge_id)
                lat = tx_delay + queue_delay + svc_delay
                eng = mec_system.compute_offload_energy(task, device_id, edge_id)
            
            episode_latencies.append(lat * 1000)  # Convert to ms
            episode_energies.append(eng)
            
            if lat > task.deadline:
                episode_violations += 1
        
        latencies.append(np.mean(episode_latencies))
        p95_latencies.append(np.percentile(episode_latencies, 95))
        energies.append(np.mean(episode_energies))
        violations.append(episode_violations)
        iterations_list.append(iters)
        times.append(elapsed)
        
        print(f"  Latency: {latencies[-1]:.2f} ms, P95: {p95_latencies[-1]:.2f} ms, "
              f"Violations: {violations[-1]}, Iters: {iters}, Time: {elapsed:.2f}s")
    
    return {
        'latency_mean': np.mean(latencies),
        'latency_std': np.std(latencies),
        'p95_mean': np.mean(p95_latencies),
        'p95_std': np.std(p95_latencies),
        'energy_mean': np.mean(energies),
        'violations': np.sum(violations),
        'iterations_mean': np.mean(iterations_list),
        'iterations_std': np.std(iterations_list),
        'time_mean': np.mean(times),
        'utility_mean': np.mean(utilities_list),
        'latencies': latencies,
        'p95_latencies': p95_latencies
    }


def plot_results(quantum_results: Dict, classical_results: Dict):
    """Plot comparison results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Latency comparison
    ax = axes[0, 0]
    methods = ['Best-Response', 'Quantum']
    means = [classical_results['latency_mean'], quantum_results['latency_mean']]
    stds = [classical_results['latency_std'], quantum_results['latency_std']]
    
    ax.bar(methods, means, yerr=stds, capsize=5, alpha=0.7, color=['#ff7f0e', '#2ca02c'])
    ax.set_ylabel('Mean Latency (ms)')
    ax.set_title('Mean Latency Comparison')
    ax.grid(axis='y', alpha=0.3)
    
    # P95 latency
    ax = axes[0, 1]
    means = [classical_results['p95_mean'], quantum_results['p95_mean']]
    stds = [classical_results['p95_std'], quantum_results['p95_std']]
    
    ax.bar(methods, means, yerr=stds, capsize=5, alpha=0.7, color=['#ff7f0e', '#2ca02c'])
    ax.set_ylabel('P95 Latency (ms)')
    ax.set_title('P95 Latency Comparison')
    ax.grid(axis='y', alpha=0.3)
    
    # Convergence iterations
    ax = axes[1, 0]
    means = [classical_results['iterations_mean'], quantum_results['iterations_mean']]
    stds = [classical_results['iterations_std'], quantum_results['iterations_std']]
    
    ax.bar(methods, means, yerr=stds, capsize=5, alpha=0.7, color=['#ff7f0e', '#2ca02c'])
    ax.set_ylabel('Iterations to Converge')
    ax.set_title('Convergence Speed Comparison')
    ax.grid(axis='y', alpha=0.3)
    
    # Latency distributions
    ax = axes[1, 1]
    ax.boxplot([classical_results['latencies'], quantum_results['latencies']], 
                labels=methods, patch_artist=True)
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Latency Distribution')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('quantum_mec_results.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'quantum_mec_results.png'")
    plt.show()


def print_summary_table(results_dict: Dict[str, Dict]):
    """Print formatted results table"""
    
    print("\n" + "="*80)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("="*80)
    print(f"{'Method':<20} {'Latency (ms)':<15} {'P95 (ms)':<15} {'Iters':<10} {'Violations':<12}")
    print("-"*80)
    
    for method, results in results_dict.items():
        print(f"{method:<20} "
              f"{results['latency_mean']:>6.1f}±{results['latency_std']:>4.1f}    "
              f"{results['p95_mean']:>6.1f}±{results['p95_std']:>4.1f}    "
              f"{results['iterations_mean']:>5.1f}±{results['iterations_std']:>3.1f}  "
              f"{results['violations']:>4}")
    
    print("-"*80)
    
    # Calculate improvements
    if 'Quantum' in results_dict and 'Best-Response' in results_dict:
        quantum = results_dict['Quantum']
        classical = results_dict['Best-Response']
        
        lat_improve = 100 * (classical['latency_mean'] - quantum['latency_mean']) / classical['latency_mean']
        p95_improve = 100 * (classical['p95_mean'] - quantum['p95_mean']) / classical['p95_mean']
        iter_improve = 100 * (classical['iterations_mean'] - quantum['iterations_mean']) / classical['iterations_mean']
        
        print(f"\nQuantum Improvements over Classical:")
        print(f"  Mean Latency: {lat_improve:>6.1f}%")
        print(f"  P95 Latency:  {p95_improve:>6.1f}%")
        print(f"  Convergence:  {iter_improve:>6.1f}%")
    
    print("="*80 + "\n")


def gamma_ablation_study(n_devices: int = 8):
    """Study effect of entanglement level gamma"""
    
    print("\n" + "="*60)
    print("GAMMA ABLATION STUDY")
    print("="*60 + "\n")
    
    gamma_values = [0, np.pi/8, np.pi/4, 3*np.pi/8, np.pi/2]
    latencies = []
    utilities = []
    
    mec_system = MECSystem(n_devices=n_devices, n_edges=2)
    
    for gamma in gamma_values:
        print(f"\nTesting gamma = {gamma:.4f} ({gamma/np.pi:.2f}π)")
        
        episode_lats = []
        episode_utils = []
        
        for episode in range(10):
            tasks = mec_system.generate_tasks()
            mec_system.channel_gains = np.random.rayleigh(1.0, (n_devices, 2))
            
            solver = QuantumNashSolver(
                n_devices=n_devices,
                gamma_min=gamma,
                gamma_max=gamma,
                max_iters=30,
                n_shots=200
            )
            
            params, results = solver.solve(mec_system, tasks, verbose=False)
            prob_dist = results['prob_dist']
            state_idx = np.random.choice(len(prob_dist), p=prob_dist)
            actions = solver.decode_actions(state_idx)
            
            utils = mec_system.compute_utility(tasks, actions)
            episode_utils.append(np.mean(utils))
            
            # Compute latency
            lats = []
            for dev_id, (task, action) in enumerate(zip(tasks, actions)):
                if action == 0:
                    lat = mec_system.compute_local_delay(task, dev_id)
                else:
                    edge_id = min(action - 1, 1)
                    lat = (mec_system.compute_transmission_delay(dev_id, edge_id, task.data_size) +
                           mec_system.compute_queueing_delay(edge_id) +
                           mec_system.compute_service_delay(task, edge_id))
                lats.append(lat * 1000)
            
            episode_lats.append(np.mean(lats))
        
        latencies.append(np.mean(episode_lats))
        utilities.append(np.mean(episode_utils))
        
        print(f"  Mean Latency: {latencies[-1]:.2f} ms")
        print(f"  Mean Utility: {utilities[-1]:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    gamma_labels = [f'{g/np.pi:.2f}π' for g in gamma_values]
    plt.plot(gamma_labels, latencies, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Entanglement Level γ')
    plt.ylabel('Mean Latency (ms)')
    plt.title('Latency vs Entanglement Level')
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(gamma_labels, utilities, 's-', linewidth=2, markersize=8, color='green')
    plt.xlabel('Entanglement Level γ')
    plt.ylabel('Mean Utility')
    plt.title('Utility vs Entanglement Level')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gamma_ablation.png', dpi=300, bbox_inches='tight')
    print("\nGamma ablation plot saved as 'gamma_ablation.png'")
    plt.show()
    
    return gamma_values, latencies, utilities


def main():
    """Main execution function"""
    
    print("\n" + "="*80)
    print(" "*20 + "QUANTUM MEC RESOURCE ALLOCATION")
    print(" "*15 + "Comprehensive Experimental Evaluation")
    print("="*80)
    
    # Configuration
    N_DEVICES = 8
    N_EDGES = 2
    N_EPISODES = 20
    
    # 1. Main comparison
    print("\n[1/3] Running main method comparison...")
    quantum_results = run_experiment(
        n_devices=N_DEVICES,
        n_edges=N_EDGES,
        n_episodes=N_EPISODES,
        method='quantum'
    )
    
    classical_results = run_experiment(
        n_devices=N_DEVICES,
        n_edges=N_EDGES,
        n_episodes=N_EPISODES,
        method='classical'
    )
    
    # Print summary
    results_dict = {
        'Best-Response': classical_results,
        'Quantum': quantum_results
    }
    print_summary_table(results_dict)
    
    # Plot comparison
    plot_results(quantum_results, classical_results)
    
    # 2. Gamma ablation
    print("\n[2/3] Running gamma ablation study...")
    gamma_vals, lats, utils = gamma_ablation_study(n_devices=N_DEVICES)
    
    # 3. Save results
    print("\n[3/3] Saving results...")
    results_data = {
        'quantum': quantum_results,
        'classical': classical_results,
        'gamma_ablation': {
            'gamma_values': [float(g) for g in gamma_vals],
            'latencies': [float(l) for l in lats],
            'utilities': [float(u) for u in utils]
        },
        'config': {
            'n_devices': N_DEVICES,
            'n_edges': N_EDGES,
            'n_episodes': N_EPISODES
        }
    }
    
    with open('experiment_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print("\nResults saved to 'experiment_results.json'")
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - quantum_mec_results.png")
    print("  - gamma_ablation.png")
    print("  - experiment_results.json")
    print("\n")


if __name__ == "__main__":
    main()
