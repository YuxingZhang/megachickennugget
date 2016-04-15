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

/* TODO update z_dn */
bool UpdateZ(int d, int n, vector<mat>& z, mat& eta_m, mat& rho_m, mat& rho_s, mat& xi_KW, mat& alpha_K,
        vector< vector<string> >& W, map<string, int>& word2idx, int K, int V, double EPS) {
    bool converge = true;
    vec z_dn_old = z[d].row(n);

    double temp = 0.0;
    for (int k = 0; k < K; k++) {
        double E1 = eta_m(d, k);
        for (int w = 0; w < V; w++) {
            temp += - lambda(xi_KW(k, w)) * (rho_s(k, w) + pow(rho_m(k, w), 2)) - (0.5 - 2.0 * alpha_K(k) * lambda(xi_KW(k, w)) * rho_m(k, w))
                + xi_KW(k, w) / 2.0 - lambda(xi_KW(k, w)) * (pow(alpha_K(k), 2) - pow(xi_KW(k, w), 2)) - log(1.0 + exp(xi_KW(k, w)));
        }

        string w_dn = W[d][n];
        double E2 = rho_m(k, word2idx[w_dn]) + alpha_K(k) * (V / 2.0 - 1.0) + temp;
        z[d](n, k) = exp(E1 + E2);
    }
    z[d].row(n) = normalise(z[d].row(n), 1);

    for (int k = 0; k < K; k++) {
        if (abs(z[d](n, k) - z_dn_old(k)) / abs(z_dn_old(k)) > EPS) {
            converge = false;
            break;
        }
    }

    return converge;
}

/* TODO update auxiliary */
void UpdateAuxiliary(int idx, vec& alpha, mat& xi, mat& mean, mat& sd, int sum_idx) {
    double temp1 = 0;
    double temp2 = 0;

    for (int i = 0; i < sum_idx; i++) {
        temp2 += lambda(xi(idx, i));
        temp1 += lambda(xi(idx, i)) * mean(idx, i);
        xi(idx, i) = sqrt(sd(idx, i) + pow(mean(idx, i), 2) - 2 * alpha(idx) * mean(idx, i) + pow(alpha(idx), 2));
    }
    alpha(idx) = (0.5 * (sum_idx / 2.0 - 1.0) + temp1) / temp2;
    return;
}

/* update eta */
bool UpdateEta(int d, mat& eta_m, mat& eta_s, mat& xi_DK, vec& alpha_D, mat& u_m, mat& a_m, vector<mat>& z, double gamma, vector<int>& N, int K, double EPS){
    bool converge = true;
    double temp;

    vec mu_old = eta_m.row(d);
    vec sigma_old = eta_s.row(d);

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

    vec mu_old = a_m.row(d);
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
        if(abs(a_m(d, k) - mu_old(k)) / abs(mu_old(k)) > EPS){
            converge = false;
            break;
        }
    }
    if(converge){
        /*
        if((abs(a_s - sigma_old) / abs(sigma_old)).max() > EPS){
        }
        */
            converge = false;
    }
    return converge;
}

/* update rho */
bool UpdateRho(int k, mat& rho_m, mat& rho_s, vector<mat>& z, mat& up_m, vec& alpha_K, mat& xi_KW, mat& word_embedding, vector<vector<string> >& W, map<int, string>& idx2word, double beta, int D, vector<int>& N, int V, double EPS){
    bool converge = true;
    double c_kw, m_k;

    vec mu_old = rho_m.row(k);
    vec sigma_old = rho_s.row(k);

    for(int w = 0; w < V; w++){
        c_kw = 0;
        m_k = 0;
        for(int d = 0; d < D; d++){
            for(int n = 0; n < N[d]; n++){
                if(!W[d][n].compare(idx2word[w])){
                    c_kw += (z[d])(n, k);
                }
                m_k += (z[d])(n, k);
            }
        }
        rho_s(k, w) = 1.0 / (beta + 2 * m_k * lambda(xi_KW(k, w)));
        rho_m(k, w) = beta * dot(word_embedding.row(w), up_m.row(k)) + c_kw - m_k * (0.5 - 2 * alpha_K(k) * lambda(xi_KW(k, w)));
        rho_m(k, w) *= rho_s(k, w);
    }

    for(int w = 0; w < V; w++){
        if(max(abs(rho_m(k, w) - mu_old(w)) / abs(mu_old(w)), abs(rho_s(k, w) - sigma_old(w)) / abs(sigma_old(w))) > EPS){
            converge = false;
            break;
        }
    }
    return converge;
}

/* update u_prime */
bool UpdateUp(int k, mat& up_m, mat& up_s, mat& rho_m, mat& word_embedding, double beta, int WORD_DIM, int V, double EPS){
    bool converge = true;
   
    vec mu_old = up_m.row(k);
    vec temp(WORD_DIM, fill::zeros);
    for(int w = 0; w < V; w++){
        temp += word_embedding.row(w) * rho_m(k, w);
    }
    up_m.row(k) = beta * (up_s * temp);

    for(int w = 0; w < V; w++){
        if(abs(up_m(k, w) - mu_old(w)) / abs(mu_old(w)) > EPS){
            converge = false;
            break;
        }
    }
    return converge;
}

/* update l */
double UpdateL(mat& up_m, mat& up_s, int WORD_DIM, int K){
    double temp = 0;

    for(int k = 0; k < K; k++){
        temp += dot(up_m.row(k), up_m.row(k));
    }
    return K * WORD_DIM / (temp + K * trace(up_s));
}

/* update kappa */
double UpdateKappa(mat& u_m, mat& u_s, int DOC_DIM, int K){
    double temp = 0;

    for(int k = 0; k < K; k++){
        temp += dot(u_m.row(k), u_m.row(k));
    }
    return K * DOC_DIM / (temp + K * trace(u_s));
}

/* update c */
double UpdateC(mat& a_m, mat& a_s, int DOC_DIM, int D){
    double temp = 0;

    for(int d = 0; d < D; d++){
        temp += dot(a_m.row(d), a_m.row(d));
    }
    return D * DOC_DIM / (temp + D * trace(a_s));
}

/* update beta */
double UpdateBeta(mat& up_m, mat& up_s, mat& word_embedding, mat& rho_m, mat& rho_s, int V, int K){
    double temp = 0;

    for(int k = 0; k < K; k++){
        for(int w = 0; w < V; w++){
            temp += (trace((up_m.row(k).t() * up_m.row(k) + up_s) * (word_embedding.row(w).t() * word_embedding.row(w))) + pow(rho_m(k, w), 2) + rho_s(k, w) - 2 * dot(word_embedding.row(w), up_m.row(k)) * rho_m(k, w));
        }
    }
    return K * V / temp;
}
    
/* update gamma */
double UpdateGamma(mat& eta_m, mat& eta_s, mat& a_m, mat& a_s, mat& u_m, mat& u_s, int D, int K){
    double temp = 0;
    for(int d = 0; d < D; d++){
        for(int k = 0; k < K; k++){
            temp += (pow(eta_m(d, k), 2) + eta_s(d, k) - 2 * eta_m(d, k) * dot(u_m.row(k), a_m.row(d)) + trace((a_m.row(d).t() * a_m.row(d) + a_s) * (u_m.row(k).t() * u_m.row(k) + u_s)));
        }
    }
   return D * K / temp;
}

