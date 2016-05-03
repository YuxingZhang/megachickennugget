double ElboZ(set<int>& idx_set, vector<int>& N, vector<mat>& z, mat& gamma, mat& lambda, vector< vector<string> >& W, map<string, int>& word2idx, int K, int V) {
    double total = 0.0;
    for (set<int>::iterator iter = idx_set.begin(); iter != idx_set.end(); iter++) {
        int d = *iter;
        for (int n = 0; n < N[d]; n++) {
            for (int k = 0; k < K; k++) {
                double elbo = 0;
                //elbo += z[d](n, k) * (digamma(lambda(k, word2idx[W[d][n]])) - digamma(sum(lambda.row(k)))
                //        + digamma(gamma(d, k)) - digamma(sum(gamma.row(d))) - log(z[d](n, k)));
                elbo += digamma(lambda(k, word2idx[W[d][n]]));
                elbo -= digamma(sum(lambda.row(k)));
                elbo += digamma(gamma(d, k));
                elbo -= digamma(sum(gamma.row(d)));
                elbo -= log(z[d](n, k));
                elbo *= z[d](n, k);
                total += elbo;
                /*
                if (std::isnan(elbo)) {
                    cout << "nan elbo " << d << " " << n << " " << k << " " << z[d](n, k) << endl;
                }
                */
            }
            //cout << "======================== d = " << d << " =========================" << endl;
            //cout << "======================== " << total << " =========================" << endl;
        }
    }
    return total;
}

/* update z_dn */
double UpdateZ(set<int>& idx_set, vector<int>& N, vector<mat>& z, mat& gamma, mat& lambda, vector< vector<string> >& W, map<string, int>& word2idx, int K, int V) {
    for (set<int>::iterator iter = idx_set.begin(); iter != idx_set.end(); iter++) {
        int d = *iter;
        for (int n = 0; n < N[d]; n++) {
            double max_z = -100000.0;
            for (int k = 0; k < K; k++) {
                double temp1 = 0.0;
                double temp2 = 0.0;
                for (int w = 0; w < V; w++) {
                    temp1 += lambda(k, w);
                }
                for (int l = 0; l < K; l++) {
                    temp2 += gamma(d, l);
                }
                double exponent = digamma(lambda(k, word2idx[W[d][n]])) - digamma(temp1)
                        + digamma(gamma(d, k)) - digamma(temp2);
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
        elbo += log(tgamma(gamma(d, k)));
    }
    elbo -= log(tgamma(sum(gamma.row(d))));
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
        elbo += log(tgamma(lambda(k, w)));
    }
    elbo -= log(tgamma(sum(lambda.row(k))));
    return elbo;
}

/* update Lambda */
double UpdateLambda(int k, mat& lambda, vector<mat>& z, vector<vector<string> >& W, map<string, int>& word2idx, vec& beta, int D, vector<int>& N, int V){
    double c_kw;
    double c_k[V];
    memset(c_k, 0, sizeof(c_k));

    for (int d = 0; d < D; d++){
        for (int n = 0; n < N[d]; n++){
            c_k[word2idx[W[d][n]]] += z[d](n, k);
        }
    }

    for (int w = 0; w < V; w++){
        lambda(k, w) = c_k[w] + beta(w);
    }
    return 0.0;
}

/*
void UpdateBeta() {
    return;
}
*/

void UpdateAlpha() {
    return;
}
