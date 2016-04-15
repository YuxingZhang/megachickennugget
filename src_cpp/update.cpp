double lambda(double xi){
    return 1.0 / (2 * xi) * (1.0 / (1 + exp(-xi)) - 0.5);
}

/* update U' sigma */
void ComputeUpSigma(mat& up_s, mat& word_embedding, double& beta, double& l, int WORD_DIM, int V){
    mat temp(WORD_DIM, WORD_DIM, fill::zeros);
    for(int i = 0; i < V; i++){
        temp += (word_embedding.row(i).t() * word_embedding.row(i));
    }
    up_s = (l * eye<mat>(WORD_DIM, WORD_DIM) + beta * temp).i();
    return;
}

/* update z_dn */
bool UpdateZ(int d, int n, vector<mat>& z, mat& eta_m, mat& rho_m, mat& rho_s, mat& xi_KW, mat& alpha_K,
        vector< vector<string> >& W, map<int, string>& word2idx, int K, int V, double EPS) {
    bool converge = true;
    vec z_dn_old = z[d].row(n);

    double temp = 0.0;
    for (int k = 0; k < K; k++) {
        for (int w = 0; w < V; w++) {
            temp += - lambda(xi_KW(k, w)) * (rho_s(k, w) + pow(rho_m(k, w), 2)) - (0.5 - 2.0 * alpha_K(k) * lmd(xi_KW(k, w)) * rho_m(k, w))
                + xi_KW(k, w) / 2.0 - lmd(xi_KW(k, w)) * ();
        }
    }
    return false;
}

/* update eta */
bool UpdateEta(int d, mat& eta_m, mat& eta_s, mat& xi_DK, vec& alpha_D, mat& u_m, mat& a_m, vector<mat>& z, double gamma, vector<int>& N, int K, double EPS){
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

/* update a */
bool UpdateA(int d, mat& a_m, mat& a_s, mat& u_m, mat& u_s, mat& eta_m, double c, double gamma, int DOC_DIM, int K, double EPS){
    bool converge = true;
    mat temp1(DOC_DIM, DOC_DIM, fill::zeros);
    vec temp2(DOC_DIM, fill::zeros);

    vec mu_old = a_u.row(d);
    mat sigma_old = a_s;

    if(d == 0){
        for(int k = 0; k < K; k++){
            temp1 += (u_m.row(k).t() * u_m.row(k));
            temp2 += (eta_m(d, k) * u_m.row(k).t());
        }
        a_s = (gamma * temp1 + gamma * K * u_s + c * eye(DOC_DIM, DOC_DIM)).i();
        a_m.row(d) = gamma * (a_s * temp2);
    }
    else{
        for(int k = 0; k < K; k++){
            temp2 += (eta_m(d, k) * u_m.row(k).t());
        }
        a_m.row(d) = gamma * (a_s * temp2);
    }

    for(int k = 0; k < K; k++){
        if(abs(a_u(d, k) - mu_old(k)) / abs(mu_old(k)) > EPS){
            converge = false;
            break;
        }
    }
    if(converge){
        if(max(abs(a_s - sigma_old) / abs(sigma_old)) > EPS){
            converge = false;
        }
    }
    return converge;
}
