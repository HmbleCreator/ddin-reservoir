import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from pyNN.brian2 import setup, run, end, Population, IF_cond_exp, SpikeSourceArray, Projection, AllToAllConnector, STDPMechanism, SpikePairRule, AdditiveWeightDependence, NumpyRNG, FixedProbabilityConnector, StaticSynapse

def evaluate_snn(w_in, tau_max, norm_opt, labels):
    print(f"\nIntegrating: w_in={w_in:.3f}, tau_max={tau_max:.1f}ms")
    n_roots = len(labels)
    in_dim = norm_opt.shape[1]
    
    min_val = norm_opt.min()
    max_val = norm_opt.max()
    rates = ((norm_opt - min_val) / (max_val - min_val)) * 100.0
    
    time_per_root = 100.0
    total_time = n_roots * time_per_root
    dt = 1.0

    setup(timestep=dt, min_delay=1.0, max_delay=10.0)

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
    
    res_size = 128
    np.random.seed(42)
    tau_m_dist = np.random.uniform(10.0, tau_max, size=res_size)
    tau_syn_E_dist = np.random.uniform(2.0, 10.0, size=res_size)
    tau_syn_I_dist = np.random.uniform(2.0, 10.0, size=res_size)
    
    reservoir = Population(res_size, IF_cond_exp(tau_m=tau_m_dist, tau_syn_E=tau_syn_E_dist, tau_syn_I=tau_syn_I_dist, v_rest=-65.0, v_reset=-65.0, v_thresh=-50.0), label="Reservoir")
    
    in_proj = Projection(input_pop, reservoir, AllToAllConnector(), StaticSynapse(weight=w_in, delay=1.0))
    
    stdp_model = STDPMechanism(
        timing_dependence=SpikePairRule(tau_plus=20.0, tau_minus=20.0, A_plus=0.01, A_minus=0.012),
        weight_dependence=AdditiveWeightDependence(w_min=0.0, w_max=0.05),
        weight=0.01,
        delay=1.0,
        dendritic_delay_fraction=0.0
    )
    rec_proj = Projection(reservoir, reservoir, FixedProbabilityConnector(p_connect=0.2, rng=NumpyRNG(seed=42)), synapse_type=stdp_model)
    
    reservoir.record(['spikes'])

    run(total_time)
    
    data = reservoir.get_data()
    spiketrains = data.segments[0].spiketrains
    
    readout_rates = np.zeros((n_roots, res_size))
    for root_idx in range(n_roots):
        start_t = root_idx * time_per_root
        end_t = (root_idx + 1) * time_per_root
        for n_i, st in enumerate(spiketrains):
            spikes_in_window = st.magnitude[(st.magnitude >= start_t) & (st.magnitude < end_t)]
            readout_rates[root_idx, n_i] = len(spikes_in_window) / (time_per_root / 1000.0)

    end()
    
    if readout_rates.sum() == 0:
        return 0.0, readout_rates
         
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    norm_readout = StandardScaler().fit_transform(readout_rates)
    
    le = LabelEncoder()
    ids = le.fit_transform(labels)
    
    best_ari_axis = -1
    for k in range(5, 9):
        pred = KMeans(n_clusters=k, random_state=42, n_init=15).fit_predict(norm_readout)
        a = adjusted_rand_score(ids, pred)
        if a > best_ari_axis:
            best_ari_axis = a
            
    return best_ari_axis, norm_readout

def main():
    print("="*65)
    print("DDIN v17b -- SPNN Tuning Grid")
    print("="*65)

    try:
        norm_opt = np.load('ddin_v16_norm_codes.npy')
        labels = np.load('ddin_v16_labels_axis.npy')
    except Exception as e:
        print("Data files missing.")
        return
    
    configs = [
        (0.04, 200.0),    # Double input wt, 200ms context
        (0.06, 500.0),    # Strong drive, deep 500ms context
        (0.08, 1000.0),   # Very strong drive, extreme 1000ms context
    ]

    best_ari = -1
    best_conf = None
    best_readout = None
    
    for w_in, tau_max in configs:
        ari, readout = evaluate_snn(w_in, tau_max, norm_opt, labels)
        print(f"--> RESULT: w_in={w_in:.3f}, tau_max={tau_max:.1f}ms produces ARI: {ari:.4f}")
        
        if ari > best_ari:
            best_ari = ari
            best_conf = (w_in, tau_max)
            best_readout = readout

    print(f"\n{'='*65}")
    print(f"BEST CONFIGURATION: w_in={best_conf[0]:.3f}, tau_max={best_conf[1]:.1f}ms")
    print(f"BEST ARI (axis): {best_ari:.4f}")
    if best_ari > 0.0366:
         print(">>> BREAKTHROUGH: SNN EXCEEDED STATIC BASELINE OF 0.0366!")
    else:
         print(">>> Still below 0.0366 baseline. Further tuning required.")
    print(f"{'='*65}")
    
    if best_readout is not None:
        np.save('ddin_v17b_best_snn_readout.npy', best_readout)

if __name__ == '__main__':
    main()
