double lambda(double xi){
    return 1.0 / (2 * xi) * (1.0 / (1 + exp(-xi)) - 0.5);
}

/* update U' sigma */
void ComputeUpSigma(mat& up_s, mat& word_embedding, double& beta, double& l, int WORD_DIM, int V){
    mat temp(WORD_DIM, WORD_DIM, fill::zeros);
    for(int i = 0; i < V; i++){
        temp += (word_embedding.row(i) * word_embedding.row(i).t());
    }
    up_s = (l * eye<mat>(WORD_DIM, WORD_DIM) + beta * temp).i();
    return;
}

/* update z_dn */


/* update eta */
bool UpdateEta(int d, mat& eta_m, mat& eta_s, mat& xi_DK, vec& alpha_D, mat& u_m, mat& a_m, vector<mat>& z, double& gamma, vector<int> N, int K, double EPS){
    bool converge = true;
    double temp;

    vec mu_old, sigma_old;
    mu_old = eta_m.row(d);
    sigma_old = eta_s.row(d);

    for(int k = 0; k < K; k++){
        eta_s(d, k) = 1.0 / (gamma + 2 * N[d] * lambda(xi_DK(d, k)));
        temp = 0;
        for(int n = 0; n < N[d]; n++){
            temp += (z[d])(n, k);
        }
        eta_m(d, k) = gamma * dot(u_m.row(k), a_m.row(d)) + N[d] * (2 * alpha_D(d) * lambda(xi_DK(d, k)) - 0.5) + temp;
        eta_m(d, k) *= eta_s(d, k);
    }

    for(int k = 0; k < K; k++){
        if(max(abs(eta_m(d, k) - mu_old(k)) / abs(mu_old(k)), abs(eta_s(d, k) - sigma_old(k)) / abs(sigma_old(k))) > EPS){
            converge = false;
            break;
        }
    }

    return converge;
}

