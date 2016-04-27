#include <random>
void SampleEta(int d, mat& eta_m, mat& u_m, mat& a_m, vector<mat>& z, double gamma, int N, int K){
	// initialize the value of eta
	vec prev_eta = eta_m.row(d);
    vec eta_sample_sum(K, fill::zeros);
    double new_eta, C, mu, sigma, accept;
	std::default_random_engine generator;
    int iterations = 1000;
    int num_sample = 100;

	// proposal distribution
    double proposal_mu[K];
    for (int k = 0; k < K; k++) {
        proposal_mu[k] = dot(u_m.row(k), a_m.row(d)) + z[d](n, k) / gamma;
    }
    double proposal_sigma = 1 / gamma;

	// Gibbs sampling
    for (int i = 0; i < iterations; i++){
        for (int k = 0; k < K; k++) {
            normal_distribution<double> distribution(proposal_mu, proposal_sigma);
            // propose from the normal distribution
            new_eta = distribution(generator);
            // compute the rejection criteria
            accept = min((C + exp(new_eta)) / (C + exp(prev_eta(k))), 1);
            // get the new sample
            uniform_real_distribution<double> distribution(0.0, 1.0);
            double prob = distribution(generator);
            if(prob > accept) new_eta = prev_eta(k);
            
            if(i > iterations - num_sample){
                eta_sample_sum(k) += new_eta;
            }
            prev_eta(k) = new_eta;
        }
    }
    eta_m.row(d) = eta_sample_sum.t() / num_sample;

	return;
}
