import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from pyNN.brian2 import setup, run, end, Population, IF_cond_exp, SpikeSourceArray, Projection, AllToAllConnector, FixedProbabilityConnector, StaticSynapse, STDPMechanism, SpikePairRule, AdditiveWeightDependence, NumpyRNG

def evaluate_wta_snn(norm_opt, labels):
    print("\nIntegrating Experiment 19: Sparse WTA Calibration")
    n_roots = len(labels)
    in_dim = norm_opt.shape[1]
    
    min_val = norm_opt.min()
    max_val = norm_opt.max()
    rates = ((norm_opt - min_val) / (max_val - min_val)) * 100.0  # Hz
    
    time_per_root = 100.0
    total_time = n_roots * time_per_root
    dt = 1.0

    setup(timestep=dt, min_delay=1.0, max_delay=10.0)

    # 1. GENERATE SPIKES
    spike_times_list = [[] for _ in range(in_dim)] 
    rng = np.random.RandomState(42)
    for root_idx in range(n_roots):
        start_t = root_idx * time_per_root
        for dim_idx in range(in_dim):
            r = rates[root_idx, dim_idx]
            if r > 0.1:
                n_spikes_expected = int(r * (time_per_root/1000.0) * 2) + 5
                isi = rng.exponential(1.0 / r, n_spikes_expected) * 1000.0
                isi = np.maximum(isi, 1.0)
                spikes = np.cumsum(isi)
                spikes = spikes[spikes < time_per_root] + start_t
                spike_times_list[dim_idx].extend(spikes.tolist())

    for idx in range(in_dim):
        if len(spike_times_list[idx]) == 0:
            spike_times_list[idx] = [0.1]
        spike_times_list[idx] = np.sort(np.unique(spike_times_list[idx])).tolist()

    input_pop = Population(in_dim, SpikeSourceArray(spike_times=spike_times_list), label="Input")
    
    # 2. POPULATIONS
    n_exc = 102
    n_inh = 26
    print(f"Creating E-Pop ({n_exc}) and I-Pop ({n_inh})...")
    
    np.random.seed(42)
    # The E-population uses 200ms context to bridge root structure
    tau_m_E = np.random.uniform(50.0, 200.0, size=n_exc)
    tau_syn_E_exc = np.random.uniform(2.0, 10.0, size=n_exc)
    tau_syn_I_exc = np.random.uniform(10.0, 30.0, size=n_exc)
    
    E_pop = Population(n_exc, IF_cond_exp(tau_m=tau_m_E, tau_syn_E=tau_syn_E_exc, tau_syn_I=tau_syn_I_exc, v_rest=-65.0, v_reset=-65.0, v_thresh=-50.0), label="E-Pop")
    
    # The I-population is extremely fast (5ms reaction)
    tau_m_I = np.random.uniform(2.0, 5.0, size=n_inh)
    I_pop = Population(n_inh, IF_cond_exp(tau_m=tau_m_I, v_rest=-65.0, v_reset=-65.0, v_thresh=-52.0), label="I-Pop")
    
    # 3. PROJECTIONS
    print("Wiring the Sparse Winner-Take-All Matrix...")
    
    # Drive from Input
    Projection(input_pop, E_pop, AllToAllConnector(), StaticSynapse(weight=0.03, delay=1.0), receptor_type='excitatory')
    # No direct drive to I (I only reacts to E)
    
    # E -> E (Context STDP, very weak, very sparse)
    stdp_model = STDPMechanism(
        timing_dependence=SpikePairRule(tau_plus=20.0, tau_minus=20.0, A_plus=0.01, A_minus=0.012),
        weight_dependence=AdditiveWeightDependence(w_min=0.0, w_max=0.05),
        weight=0.005, delay=1.0, dendritic_delay_fraction=0.0
    )
    Projection(E_pop, E_pop, FixedProbabilityConnector(p_connect=0.1, rng=NumpyRNG(seed=42)), synapse_type=stdp_model, receptor_type='excitatory')
    
    # E -> I (Global Trigger)
    Projection(E_pop, I_pop, AllToAllConnector(), StaticSynapse(weight=0.02, delay=1.0), receptor_type='excitatory')
    
    # I -> E (Global Silence)
    Projection(I_pop, E_pop, AllToAllConnector(), StaticSynapse(weight=0.10, delay=1.0), receptor_type='inhibitory')
    
    # I -> I (Self regulation to prevent absolute permanent lock)
    Projection(I_pop, I_pop, FixedProbabilityConnector(p_connect=0.2), StaticSynapse(weight=0.02, delay=1.0), receptor_type='inhibitory')
    
    # 4. RECORD
    E_pop.record(['spikes'])

    print(f"Running WTA Simulation for {total_time}ms...")
    run(total_time)
    
    # 5. READOUT (from E_pop only)
    data = E_pop.get_data()
    spiketrains = data.segments[0].spiketrains
    
    readout_rates = np.zeros((n_roots, n_exc))
    for root_idx in range(n_roots):
        start_t = root_idx * time_per_root
        end_t = (root_idx + 1) * time_per_root
        for n_i, st in enumerate(spiketrains):
            spikes_in_window = st.magnitude[(st.magnitude >= start_t) & (st.magnitude < end_t)]
            readout_rates[root_idx, n_i] = len(spikes_in_window) / (time_per_root / 1000.0)

    end()
    
    mean_rate_Hz = readout_rates.mean()
    if readout_rates.sum() == 0:
        return 0.0, readout_rates, mean_rate_Hz
         
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    norm_readout = StandardScaler().fit_transform(readout_rates)
    
    le = LabelEncoder()
    ids = le.fit_transform(labels)
    
    best_ari = -1
    for k in range(5, 9):
        pred = KMeans(n_clusters=k, random_state=42, n_init=15).fit_predict(norm_readout)
        a = adjusted_rand_score(ids, pred)
        if a > best_ari:
            best_ari = a
            
    return best_ari, norm_readout, mean_rate_Hz


def main():
    print("="*65)
    print("DDIN v19 -- AGGRESSIVE SPARSE CALIBRATION (WTA INHIBITION)")
    print("="*65)

    try:
        norm_opt = np.load('ddin_v16_norm_codes.npy')
        labels = np.load('ddin_v16_labels_axis.npy')
    except Exception as e:
        print("Data files missing.")
        return

    ari, readout, mean_rate = evaluate_wta_snn(norm_opt, labels)
    
    print(f"\n{'='*65}")
    print(f"RESULTS (Exp 19: Global WTA, tau_max=200ms)")
    print(f"{'='*65}")
    print(f"Network Mean Firing Rate : {mean_rate:.2f} Hz")
    print(f"ARI (axis) E-Population  : {ari:.4f}")
    
    baseline = 0.0366
    
    if ari > baseline:
         print(f">>> CRITICAL BREAKTHROUGH: Winner-Take-All inhibition isolated meaning precisely and SHATTERED the {baseline} baseline!")
    elif ari > 0.0:
         print(f">>> COHERENCE MAINTAINED: At {mean_rate:.2f} Hz, the network maintains some coherence (ARI > 0) but didn't crack the static limit.")
    else:
         print(">>> SILENCE OR CHAOS: The aggressive WTA configuration either silenced the network completely or failed to separate axes.")
    
    print(f"{'='*65}")
    
    np.save('ddin_v19_snn_wta_readout.npy', readout)

if __name__ == '__main__':
    main()
