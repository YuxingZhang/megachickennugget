vec UpdateBeta(vec& beta, mat& rho_m, int V, int K){
	double NEWTON_THRESH = 0.00001;
	int MAX_ITER = 1000;
	vec result = beta;

	vec df(V, fill::zeros);
	vec g(V, fill::zeros);
	vec h(V, fill::zeros);

	// compute the first derivative
	double digamma_beta = digamma(sum(beta));
	double digamma_theta = 0;
	for(int k = 0; k < K; k++){
		digamma_theta += digamma(sum(rho_m.row(k)));
	}
	for(int w = 0; w < V; w++){
		double temp = 0;
		for(int k = 0; k < K; k++){
			temp += digamma(rho_m(k, w));
		}
		g(w) = K * (digamma_beta - digamma(beta(w))) + temp - digamma_theta;
	}
	
	// compute the Hessian
	double trigamma_beta = trigamma(sum(beta));
	for(int w = 0; w < V; w++){
		h(w) = K * trigamma(beta(w));
	}

	// compute constant terms needed for gradient
	double c = sum(g / h) / (- 1 / trigamma_beta + sum(1 / h));
	int iter = 0;
	do{
		iter++;

		for(int w = 0; w < V; w++){
			df(w) = (g(w) - c) / h(w);
		}
		
		result -= df;
	} while(iter < MAX_ITER && max(abs(df)) > NEWTON_THRESH);

	return result;
}
