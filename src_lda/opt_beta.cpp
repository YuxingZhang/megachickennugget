void UpdateBeta(vec& beta, mat& lambda, int V, int K){
	double NEWTON_THRESH = 0.00001;
	int MAX_ITER = 1000;
	double gamma = 0.001;

	vec df(V, fill::zeros);
	vec g(V, fill::zeros);
	vec h(V, fill::zeros);
	int iter = 0;
	do{
		// compute the first derivative
		double digamma_beta = digamma(sum(beta));
		double digamma_theta = 0;
		for(int k = 0; k < K; k++){
			digamma_theta += digamma(sum(lambda.row(k)));
		}
		for(int w = 0; w < V; w++){
			double temp = 0;
			for(int k = 0; k < K; k++){
				temp += digamma(lambda(k, w));
			}
			g(w) = K * (digamma_beta - digamma(beta(w))) + temp - digamma_theta;
		}
		cout << "this is g" << endl;
		cout << g.t() << endl;
		// compute the Hessian
		double trigamma_beta = trigamma(sum(beta));
		for(int w = 0; w < V; w++){
			h(w) = K * trigamma(beta(w));
		}

	cout << "this is h" << endl;
	cout << h.t() << endl;
		// compute constant terms needed for gradient
		double c = sum(g / h) / (- 1 / trigamma_beta + sum(1 / h));

		for(int w = 0; w < V; w++){
			df(w) = (g(w) - c) / h(w);
		}
		
		beta -= gamma * df;
		iter++;
		cout << "iteration: " << iter << endl;
		cout << beta.t() << endl;
	} while(iter < MAX_ITER && max(abs(df)) > NEWTON_THRESH);

	return;
}
