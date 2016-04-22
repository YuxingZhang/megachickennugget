void UpdateBeta(vec& beta, vec& rho_m, int K, int V){
	double eps = 0.00001;
	int MAX_ITER = 1000;

	vec df(V, fill::zeros);
	vec h(V, fill::zeros);
	vec g(V, fill::zeros);
	vec st(K, fill::zeros);
	
	for(int k = 0; k < K; k++){
		st(k) = sum(rho_m.row(k));
	}
	int iter = 0;
	double c, z, sb;
	sb = sum(beta);
	do{
		double temp1 = 0, temp2 = 0;
		z = ComputeZ();
		for(int i = 0; i < V; i++){
			g(i) = ComputeG();
			h(i) = ComputeH();
			temp1 += 1 / h(i);
			temp2 += g(i) / h(i);
		}
		c = temp2 / (1 / z + temp1);

		for(int i = 0; i < V; i++){
			df(i) = (g(i) - c) / h(i);
		}

		beta -= df;
	} while(iter < MAX_ITER && max(abs(df)) > eps);

	return;
}

double ComputeH(double beta_w, int K){
	return K * trigamma(beta_w);
}

double ComputeG(double sb, double st_k, double beta_w, double theta_kw, int K)