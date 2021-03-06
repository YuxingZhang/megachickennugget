double lambda(double xi){
    return 1.0 / (2 * xi) * (1.0 / (1 + exp(-xi)) - 0.5);
}

/* update z_dn */
bool UpdateZ(set<int>& idx_set, vector<int>& N, vector<mat>& z, mat& eta_m, mat& rho_m, vector< vector<string> >& W, map<string, int>& word2idx, int K, double EPS) {
    bool converge = true;
    double E22[K];
    for (int k = 0; k < K; k++) {
        double digamma_input = sum(rho_m.row(k));
        if (digamma_input == 0) {
            cout << "digamma 1 input = 0" << endl;
        }
        E22[k] = digammal(digamma_input);
    }
    for (set<int>::iterator iter = idx_set.begin(); iter != idx_set.end(); iter++) {
        int d = *iter;
        double E1[K];
        for (int k = 0; k < K; k++) {
            E1[k] = eta_m(d, k);
        }
        for (int n = 0; n < N[d]; n++) {
            vec z_dn_old = z[d].row(n).t();
            string w_dn = W[d][n];

            double max_p = -100000.0;
            for (int k = 0; k < K; k++) {
                double digamma_input = rho_m(k, word2idx[w_dn]);
                if (digamma_input == 0) {
                    cout << "digamma 2 input = 0" << endl;
                }
                double E2 = digammal(digamma_input) - E22[k];
                z[d](n, k) = E1[k] + E2;
                if (z[d](n, k) > max_p) {
                    max_p = z[d](n, k);
                }
            }
            for (int k = 0; k < K; k++) {
                z[d](n, k) = exp(z[d](n, k) - max_p);
            }
            z[d].row(n) = normalise(z[d].row(n), 1);

            for (int k = 0; k < K; k++) {
                if (abs(z[d](n, k) - z_dn_old(k)) / abs(z_dn_old(k)) > EPS) {
                    converge = false;
                    break;
                }
            }
        }
    }
    return converge;
}

/* update auxiliary */
void UpdateAuxiliary(int idx, vec& alpha, mat& xi, mat& mu, mat& sd, int sum_idx) {
    double temp1 = 0;
    double temp2 = 0;

    for (int i = 0; i < sum_idx; i++) {
        double temp = lambda(xi(idx, i));
        double temp_mu = mu(idx, i);
        temp2 += temp;
        temp1 += temp * temp_mu;
        xi(idx, i) = sqrt(sd(idx, i) + pow(temp_mu, 2) - 2 * alpha(idx) * temp_mu + pow(alpha(idx), 2));
    }
    alpha(idx) = (0.5 * (sum_idx / 2.0 - 1.0) + temp1) / temp2;
    return;
}

/* update eta */
bool UpdateEta(int d, mat& eta_m, mat& eta_s, mat& xi_DK, vec& alpha_D, mat& u_m, mat& a_m, vector<mat>& z, double gamma, vector<int>& N, int K, double EPS){
    bool converge = true;
    double temp;

    vec mu_old = eta_m.row(d).t();
    vec sigma_old = eta_s.row(d).t();

    for(int k = 0; k < K; k++) {
        double lambda_xi = lambda(xi_DK(d, k));
        eta_s(d, k) = 1.0 / (gamma + 2 * N[d] * lambda_xi);
        temp = 0; // sum of z_dn_k
        for(int n = 0; n < N[d]; n++) {
            temp += (z[d])(n, k);
        }
        eta_m(d, k) = gamma * dot(u_m.row(k), a_m.row(d)) + N[d] * (2 * alpha_D(d) * lambda_xi - 0.5) + temp;
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

    vec mu_old = a_m.row(d).t();
    mat sigma_old = a_s;

    if(d == 0) {
        for(int k = 0; k < K; k++) {
            temp1 += (u_m.row(k).t() * u_m.row(k));
            temp2 += (eta_m(d, k) * u_m.row(k).t());
        }
        a_s = (gamma * temp1 + gamma * K * u_s + c * eye(DOC_DIM, DOC_DIM)).i();
        a_m.row(d) = gamma * (a_s * temp2).t();

        /* Check if covariance matrix is symmetric */
        for(int k1 = 0; k1 < DOC_DIM; k1++) {
            for(int k2 = k1; k2 < DOC_DIM; k2++) {
                if (abs(a_s(k1, k2) - a_s(k2, k1)) > 0.000001) {
                    cout << "CAUTION: Covariance matrix should be symmetric (A)!!" << endl;
                }
            }
        }

    } else {
        for(int k = 0; k < K; k++) {
            temp2 += (eta_m(d, k) * u_m.row(k).t());
        }
        a_m.row(d) = gamma * (a_s * temp2).t();
    }

    mat dif = abs(a_s - sigma_old) / abs(sigma_old);
    if(dif.max() > EPS) {
        converge = false;
    } else {   
        for(int k = 0; k < DOC_DIM; k++) {
            if(abs(a_m(d, k) - mu_old(k)) / abs(mu_old(k)) > EPS) {
                converge = false;
                break;
            }
        }
    }

    return converge;
}

/* update rho */
// bool UpdateRho(int k, mat& rho_m, vector<mat>& z, vector<vector<string> >& W, map<int, string>& idx2word, vec& beta, int D, vector<int>& N, int V, double EPS){
bool UpdateRho(int k, mat& rho_m, vector<mat>& z, vector<vector<string> >& W, map<string, int>& word2idx, vec& beta, int D, vector<int>& N, int V, double EPS){
    bool converge = true;
    double c_kw;

    vec mu_old = rho_m.row(k).t();

    double c_k[V];
    memset(c_k, 0, sizeof(c_k));

    for (int d = 0; d < D; d++){
        for (int n = 0; n < N[d]; n++){
            c_k[word2idx[W[d][n]]] += z[d](n, k);
        }
    }

    // for(int w = 0; w < V; w++){
    //     c_kw = 0.0;
    //     for(int d = 0; d < D; d++){
    //         for(int n = 0; n < N[d]; n++){
    //             if(!W[d][n].compare(idx2word[w])){
    //                 c_kw += (z[d])(n, k);
    //             }
    //         }
    //     } 
    //     rho_m(k, w) = c_kw + beta(w);
    // }

    for (int w = 0; w < V; w++){
        rho_m(k, w) = c_k[w] + beta(w);
    }

    for(int w = 0; w < V; w++){
        if(abs(rho_m(k, w) - mu_old(w)) / abs(mu_old(w)) > EPS){
            converge = false;
            break;
        }
    }
    return converge;
}

/* update u */
bool UpdateU(int k, mat& u_m, mat& u_s, mat& a_m, mat& a_s, mat& eta_m, double kappa, double gamma, int DOC_DIM, int D, double EPS) {
    bool converge = true;

    vec mu_old = u_m.row(k).t();
    mat sigma_old = u_s;

    if (k == 0) {
        mat temp1(DOC_DIM, DOC_DIM, fill::zeros);
        for (int d = 0; d < D; d++) {
            temp1 += (a_m.row(d).t() * a_m.row(d));
        }
        u_s = (kappa * mat(DOC_DIM, DOC_DIM, fill::eye) + gamma * D * a_s + gamma * temp1).i();
        for(int k1 = 0; k1 < DOC_DIM; k1++){
            for(int k2 = k1; k2 < DOC_DIM; k2++){
                if (abs(u_s(k1, k2) - u_s(k2, k1)) > 0.000001){
                    cout << "CAUTION: Covariance matrix should be symmetric (U)!!" << endl;
                }
            }
        }
    }
    vec temp2(DOC_DIM, fill::zeros);
    for (int d = 0; d < D; d++) {
        temp2 += (eta_m(d, k) * a_m.row(d).t()); // col vec
    }
    u_m.row(k) = gamma * (u_s * temp2).t();

    mat dif = abs(u_s - sigma_old) / abs(sigma_old);
    if (dif.max() > EPS) {
        converge = false;
    }
    else{
        for (int d = 0; d < DOC_DIM; d++) {
            if (abs(u_m(k, d) - mu_old(d)) / abs(mu_old(d)) > EPS) {
                converge = false;
                break;
            }
        }
    }

    return converge;
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
    
/* update gamma */
double UpdateGamma(mat& eta_m, mat& eta_s, mat& a_m, mat& a_s, mat& u_m, mat& u_s, int D, int K){
    double temp = 0;
    for(int d = 0; d < D; d++){
        mat cov = a_m.row(d).t() * a_m.row(d) + a_s;
        for(int k = 0; k < K; k++){
            temp += (pow(eta_m(d, k), 2) + eta_s(d, k) - 2 * eta_m(d, k) * dot(u_m.row(k), a_m.row(d)) + trace(cov * (u_m.row(k).t() * u_m.row(k) + u_s)));
        }
    }
   return D * K / temp;
}

