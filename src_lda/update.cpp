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

void UpdateAlpha_gradient(vec& alpha, mat& gamma, int K, int D){
    double NEWTON_THREASH = 0.00001;
    int MAX_ITER = 1000;
    double step_size = 0.01;
    
    vec g(K, fill::zeros);
    vec gamma_s(D, fill::zeros);
    double sum_alpha;
    for (int d = 0; d < D; d++){
        gamma_s(d) = sum(gamma.row(d));
    }

    do{
        sum_alpha = sum(alpha);
        for (int i = 0; i < K; i++){
            double tmp = 0.0;
            for (int d = 0; d < D; d++){
                tmp += digamma(gamma(d, i)) - digamma(gamma_s(d));
            }
            g(i) = D * (digamma(sum_alpha) - digamma(alpha(i))) + tmp;
        }
        alpha -= g * step_size;
    } while(iter < MAX_ITER && max(g) > NEWTON_THRESH)

    return;
}


void UpdateAlpha(vec& alpha, mat& gamma, int K, int D) {
    double NEWTON_THRESH = 0.00001;
    int MAX_ITER = 1000;
    double step_size = 0.001;
    double max_df = 0.0;

    double c = 0.0;
    double z = 0.0;
    double g[K];
    memset(g, 0, sizeof(g));
    double h[K];
    memset(h, 0, sizeof(h));
    double update[K];
    memset(update, 0, sizeof(update));
    
    double sum_gamma[D];
    memset(sum_gamma, 0, sizeof(sum_gamma));
    for (int d = 0; d < D; d++){
        sum_gamma [d] = sum(gamma.row(d));
    }

    int iter = 0;

    do{
        double elbo_old = 0.0;
        double elbo_new = 0.0;
        double sum_alpha = sum(alpha);
        max_df = 0.0;

        for (int i = 0; i < K; i++){
            double tmp = 0.0;
            for (int d = 0; d < D; d++){
                tmp += digamma(gamma(d, i)) - digamma(sum_gamma[d]);
            }
            g [i] = D * (digamma(sum_alpha) - digamma(alpha(i))) + tmp;
            h [i] = D * trigamma(alpha(i));
        }

        z = - trigamma(sum_alpha);

        double tmp2 = 0.0;
        double tmp3 = 0.0;
        for (int i = 0; i < K; i++){
            tmp2 += g[i] / h[i];
            tmp3 += 1 / h[i];
        }
        c = tmp2 / ((1 / z) + tmp3);

        for (int d = 0; d < D; d++){
            double tmp4 = 0.0;
            for (int k = 0; k < K; k++){
                tmp4 = - lgamma(alpha(k)) + (alpha(k) - 1) * (digamma(gamma(d, k)) - digamma(sum_gamma[d]));
            }
            elbo_old += lgamma(sum_alpha) + tmp4;
        }
        cout << "elbo before alpha update: " << elbo_old << endl;

        for (int i = 0; i < K; i++){
            update[i] = (g[i] - c) / h[i];
            alpha(i) -= update[i];
            if (abs(update[i]) > max_df){
                max_df = abs(update[i]);
            }
        }

        sum_alpha = sum(alpha);
        iter ++;
        for (int d = 0; d < D; d++){
            double tmp5 = 0.0;
            for (int k = 0; k < K; k++){
                tmp5 = - lgamma(alpha(k)) + (alpha(k) - 1) * (digamma(gamma(d, k)) - digamma(sum_gamma[d]));
            }
            elbo_new += lgamma(sum_alpha) + tmp5;
        }    
        cout << "elbo after alpha update: " << elbo_new << endl;

    } while (iter < MAX_ITER && max_df > NEWTON_THRESH);

    return;
}

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
        // compute the Hessian
        double trigamma_beta = trigamma(sum(beta));
        for(int w = 0; w < V; w++){
            h(w) = K * trigamma(beta(w));
        }

        // compute constant terms needed for gradient
        double c = sum(g / h) / (- 1 / trigamma_beta + sum(1 / h));

        for(int w = 0; w < V; w++){
            df(w) = (g(w) - c) / h(w);
        }
        
        beta -= df;
        iter++;
    } while(iter < MAX_ITER && max(abs(df)) > NEWTON_THRESH);

    return;
}

void upBeta(vec& beta, mat& lambda, int V, int K){
    double NEWTON_THRESH = 0.00001;
    int MAX_ITER = 1000;
    double step_size = 0.01;
    
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
        
        beta -= g * step_size;
        iter++;
    } while(iter < MAX_ITER && max(abs(df)) > NEWTON_THRESH);

    return;
}