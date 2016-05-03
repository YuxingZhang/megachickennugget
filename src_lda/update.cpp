double ElboZ(set<int>& idx_set, vector<int>& N, vector<mat>& z, mat& gamma, mat& lambda, vector< vector<string> >& W, map<string, int>& word2idx, int K, int V) {
    double total = 0.0;
    for (set<int>::iterator iter = idx_set.begin(); iter != idx_set.end(); iter++) {
        int d = *iter;
        for (int n = 0; n < N[d]; n++) {
            for (int k = 0; k < K; k++) {
                double elbo = 0;
                elbo += digamma(lambda(k, word2idx[W[d][n]]));
                elbo -= digamma(sum(lambda.row(k)));
                elbo += digamma(gamma(d, k));
                elbo -= digamma(sum(gamma.row(d)));
                elbo -= log(z[d](n, k));
                elbo *= z[d](n, k);
                total += elbo;
            }
        }
    }
    return total;
}

/* update z_dn */
double UpdateZ(set<int>& idx_set, vector<int>& N, vector<mat>& z, mat& gamma, mat& lambda, vector< vector<string> >& W, map<string, int>& word2idx, int K, int V) {
    double sum_lambda[K];
    for (int k = 0; k < K; k++){
        sum_lambda[k] = sum(lambda.row(k));
    }
    for (set<int>::iterator iter = idx_set.begin(); iter != idx_set.end(); iter++) {
        int d = *iter;
        double sum_gamma = sum(gamma.row(d));
        for (int n = 0; n < N[d]; n++) {
            double max_z = -100000.0;
            for (int k = 0; k < K; k++) {
                double exponent = digamma(lambda(k, word2idx[W[d][n]])) - digamma(sum_lambda[k])
                        + digamma(gamma(d, k)) - digamma(sum_gamma);
                if (exponent > max_z) {
                    max_z = exponent;
                }
                z[d](n, k) = exponent;
            }
            z[d].row(n) -= max_z;
            z[d].row(n) = exp(z[d].row(n));
            z[d].row(n) += std::exp(-200.0); // adding smoothness
            z[d].row(n) = normalise(z[d].row(n), 1);
        }
    }
    return 0.0;
}

double ElboGamma(int d, mat& gamma, vector<mat>& z, vector<int>& N, int K, mat& alpha) {
    double elbo = 0.0;
    for (int k = 0; k < K; k++) {
        elbo += (alpha(k) - gamma(d, k) + sum(z[d].col(k))) * (digamma(gamma(d, k)) - digamma(sum(gamma.row(d))));
        elbo += lgamma(gamma(d, k));
    }
    elbo -= lgamma(sum(gamma.row(d)));
    return elbo;
}

/* update Gamma */
void UpdateGamma(int d, mat& gamma, vector<mat>& z, vector<int>& N, int K, mat& alpha){
    for(int k = 0; k < K; k++) {
        double temp = 0.0;
        for(int n = 0; n < N[d]; n++) {
            temp += z[d](n, k);
        }
        gamma(d, k) = alpha(k) + temp;
    }
    return;
}

double ElboLambda(int k, mat& lambda, vector<mat>& z, vector<vector<string> >& W, map<string, int>& word2idx, vec& beta, int D, vector<int>& N, int V){
    double elbo = 0;
    vec tmp(V, fill::zeros);
    for(int d = 0; d < D; d++){
        for(int n  = 0; n < N[d]; n++){
            tmp(word2idx[W[d][n]]) += z[d](n, k);
        }
    }
    for(int w = 0; w < V; w++){
        elbo += (beta(w) - lambda(k, w) + tmp(w)) * (digamma(lambda(k, w)) - digamma(sum(lambda.row(k))));
        elbo += lgamma(lambda(k, w));
    }
    elbo -= lgamma(sum(lambda.row(k)));
    return elbo;
}

/* update Lambda */
double UpdateLambda(int k, mat& lambda, vector<mat>& z, vector<vector<string> >& W, map<string, int>& word2idx, vec& beta, int D, vector<int>& N, int V){
    double temp[V];
    memset(temp, 0, sizeof(temp));
    for (int d = 0; d < D; d++) {
        for (int n = 0; n < N[d]; n++) {
            temp[word2idx[W[d][n]]] += z[d](n, k);
        }
    }
    for (int w = 0; w < V; w++) {
        lambda(k, w) = beta(w) + temp[w];
    }
    return 0.0;
}

void UpdateBeta() {
    return;
}

void UpdateAlpha() {
    return;
}
