#include <random>
void SampleEta(int d, mat& eta_m, mat& u_m, mat& a_m, vector<mat>& z, double gamma, vector<int>& N, int K){
    // initialize the value of eta
    vec prev_eta = eta_m.row(d).t();
    vec eta_sample_sum(K, fill::zeros);
    double new_eta, sum_eta, accept, C;
    std::default_random_engine generator;
    int iterations = 1000;
    int num_sample = 100;

	int block_size = 20;
	int block_num = ceil(N[d] / block_size);
	int batch = 0;
	int num_batch = 5;
	for(int n = 0; n < num_batch; n++){
		while (batch < block_num){		
		    // proposal distribution
		    double proposal_mu[K];
			int upsize = min(N[d], (batch + 1) * block_size);
			int losize = batch * block_size;
			int block_length = upsize - losize;
		    for (int k = 0; k < K; k++) {
				double tmp = 0.0;
				for (int n = losize; n < upsize; n++) {
				    tmp += z[d](n, k); 
				}
				proposal_mu[k] = dot(u_m.row(k), a_m.row(d)) + tmp / gamma - block_length / (gamma * K);
				//cout << "dot, tmp / gamma = " << dot(u_m.row(k), a_m.row(d)) << " " << tmp / gamma << endl;
				//cout << proposal_mu[k] << " ";
		    }
		    //cout << endl;
		    double proposal_sigma = 1 / gamma;

		    // Gibbs sampling
		    sum_eta = sum(exp(prev_eta));
		    for (int i = 0; i < iterations; i++){
				for (int k = 0; k < K; k++) {
				    C = sum_eta - exp(prev_eta(k));
				    
				    std::normal_distribution<double> n_distribution(proposal_mu[k], proposal_sigma);
				    // propose from the normal distribution
				    new_eta = n_distribution(generator);
				    // compute the rejection criteria
				    //accept = min(pow((C + exp(prev_eta(k))) / (C + exp(new_eta)), N[d]), 1.0);
					accept = min(1.0, exp(block_length * (log(C + exp(prev_eta(k))) - log(C + exp(new_eta)) + new_eta / K - prev_eta(k) / K)));	
					//cout << accept << endl;	
					//cout << log(C + exp(prev_eta(k))) << " " << log(C + exp(new_eta)) << " " << new_eta << " " << prev_eta(k) << endl;
				    // get the new sample
				    uniform_real_distribution<double> u_distribution(0.0, 1.0);
				    double prob = u_distribution(generator);
				    if (prob > accept) { // reject the sample if prob > accept
				    	new_eta = prev_eta(k);
				    }
					//else cout << "zhi " << new_eta << endl;
				    if(i > iterations - num_sample) { // If after burn-in period
				        eta_sample_sum(k) += new_eta;
				    }
				    sum_eta = sum_eta - exp(prev_eta(k)) + exp(new_eta);
				    prev_eta(k) = new_eta;
				}
		    }
			batch++;
		}
	}

    //cout << "before" << endl;
    //cout << eta_m.row(d);
    eta_m.row(d) = eta_sample_sum.t() / (block_num * num_sample * num_batch);
    //cout << "after" << endl;
    //cout << eta_m.row(d);

    return;
}
