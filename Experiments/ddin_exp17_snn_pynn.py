import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from pyNN.brian2 import setup, run, end, Population, IF_cond_exp, SpikeSourceArray, Projection, AllToAllConnector, STDPMechanism, SpikePairRule, AdditiveWeightDependence, NumpyRNG, FixedProbabilityConnector, StaticSynapse

def main():
    print("="*65)
    print("DDIN v17 -- SPNN with STDP (PyNN / Brian2)")
    print("Experiment 17: Contextual Resonance using Spiking Neural Network")
    print("="*65)

    # 1. Load Data from v16 (must be the exact same vectors to compare)
    try:
        norm_opt = np.load('ddin_v16_norm_codes.npy') # (150, 23)
        labels = np.load('ddin_v16_labels_axis.npy')  # (150,)
    except Exception as e:
        print("Error: Could not load v16 arrays. Please ensure v16 scripts have run.")
        print(e)
        return

    n_roots = len(labels)
    in_dim = norm_opt.shape[1] # 23
    
    # Scale norm_opt to positive rates [0, 100] Hz for Poisson generation
    min_val = norm_opt.min()
    max_val = norm_opt.max()
    rates = ((norm_opt - min_val) / (max_val - min_val)) * 100.0 # rates in Hz
    
    # Simulation Parameters
    time_per_root = 100.0 # ms 
    total_time = n_roots * time_per_root
    dt = 1.0 # ms

    # PyNN Setup
    print("Initializing PyNN simulator with brian2 backend...")
    setup(timestep=dt, min_delay=1.0, max_delay=10.0)

    # Generate spike times for the entire simulation to simulate continuing context
    # This acts as our "Contextual Resonance" - the network does not reset between roots
    spike_times_list = [[] for _ in range(in_dim)] 
    print(f"Generating continuous spike trains for {n_roots} roots over {total_time}ms...")
    
    rng = np.random.RandomState(42)
    for root_idx in range(n_roots):
        start_t = root_idx * time_per_root
        for dim_idx in range(in_dim):
            r = rates[root_idx, dim_idx]
            if r > 0.1: # avoid infinite intervals
                # Poisson spike generation
                n_spikes_expected = int(r * (time_per_root/1000.0) * 2) + 5
                isi = rng.exponential(1.0 / r, n_spikes_expected) * 1000.0 # to ms
                isi = np.maximum(isi, 1.0) # Enforce minimum 1.0ms between spikes 
                spikes = np.cumsum(isi)
                spikes = spikes[spikes < time_per_root] + start_t
                spike_times_list[dim_idx].extend(spikes.tolist())

    # Format arrays for SpikeSourceArray
    for idx in range(in_dim):
        if len(spike_times_list[idx]) == 0:
            spike_times_list[idx] = [0.1] # Ensure at least one spike to avoid empty sequence errors in pyNN
        spike_times_list[idx] = np.sort(np.unique(spike_times_list[idx])).tolist()

    # 2. Network Construction
    print("Building Spiking Neural Network Reservoir...")
    
    input_pop = Population(in_dim, SpikeSourceArray(spike_times=spike_times_list), label="Input")
    
    # Reservoir Population
    # Vary tau_m between 10ms and 100ms exactly mimicking the v3 alpha heterogeneity
    res_size = 128
    np.random.seed(42)
    tau_m_dist = np.random.uniform(10.0, 100.0, size=res_size)
    tau_syn_E_dist = np.random.uniform(2.0, 10.0, size=res_size)
    tau_syn_I_dist = np.random.uniform(2.0, 10.0, size=res_size)
    
    reservoir = Population(res_size, IF_cond_exp(tau_m=tau_m_dist, tau_syn_E=tau_syn_E_dist, tau_syn_I=tau_syn_I_dist, v_rest=-65.0, v_reset=-65.0, v_thresh=-50.0), label="Reservoir")
    
    print("Connecting Synapses with STDP...")
    # Fixed driving weights from input to reservoir
    in_proj = Projection(input_pop, reservoir, AllToAllConnector(), StaticSynapse(weight=0.02, delay=1.0))
    
    # Recurrent STDP Mechanism
    stdp_model = STDPMechanism(
        timing_dependence=SpikePairRule(tau_plus=20.0, tau_minus=20.0, A_plus=0.01, A_minus=0.012),
        weight_dependence=AdditiveWeightDependence(w_min=0.0, w_max=0.05),
        weight=0.01,
        delay=1.0,
        dendritic_delay_fraction=0.0
    )
    rec_proj = Projection(reservoir, reservoir, FixedProbabilityConnector(p_connect=0.2, rng=NumpyRNG(seed=42)), synapse_type=stdp_model)
    
    # Record spikes for readout
    reservoir.record(['spikes'])

    # 3. Execution
    print(f"Running Spiking Simulation for {total_time}ms...")
    run(total_time)
    
    # 4. Evaluation
    print("Simulation complete. Extracting Readout Firing Rates...")
    data = reservoir.get_data()
    spiketrains = data.segments[0].spiketrains
    
    readout_rates = np.zeros((n_roots, res_size))
    for root_idx in range(n_roots):
        start_t = root_idx * time_per_root
        end_t = (root_idx + 1) * time_per_root
        
        for n_i, st in enumerate(spiketrains):
            spikes_in_window = st.magnitude[(st.magnitude >= start_t) & (st.magnitude < end_t)]
            readout_rates[root_idx, n_i] = len(spikes_in_window) / (time_per_root / 1000.0) # calculate Hz

    end()
    
    # ARI Calculation
    print("\n[EVALUATION]")
    print("Running KMeans on Reservoir Readout...")
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    
    if readout_rates.sum() == 0:
         print(">>> FAILURE: The network did not emit any spikes. Increase input weights.")
         return
         
    norm_readout = StandardScaler().fit_transform(readout_rates)
    
    le = LabelEncoder()
    ids = le.fit_transform(labels)
    
    best_ari_axis = -1
    for k in range(5, 9):
        pred = KMeans(n_clusters=k, random_state=42, n_init=15).fit_predict(norm_readout)
        a = adjusted_rand_score(ids, pred)
        if a > best_ari_axis:
            best_ari_axis = a
            
    print(f"ARI (axis) from Spiking Readout  : {best_ari_axis:.4f}")
    print(f"Baseline (v16 static framework)  : 0.0366")
    print(f"{'='*65}")
    if best_ari_axis > 0.0366:
        print(">>> BREAKTHROUGH: The SNN with Contextual Resonance shattered the baseline!")
    else:
        print(">>> Results below baseline. The network is spiking, but the STDP and time constants need tuning.")
        
    np.save('ddin_v17_snn_readout.npy', norm_readout)

if __name__ == '__main__':
    main()
