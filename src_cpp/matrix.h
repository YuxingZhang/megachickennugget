class Matrix{
public:
	double** mat;
	int n, m;

	Matrix(int m, int n, bool diagnol){
		this->m = m;
		this->n = n;
		mat = new double*[n];
		for(int i = 0; i < n; i++){
			mat[i] = new double[m];
			if(diagnol){
				memset(mat[i], 0, m * sizeof(double));
				mat[i][i] = ((double) rand() / (RAND_MAX));
			}
			else{
				for(int j = 0; j < m; j++){
					mat[i][j] = ((double) rand() / (RAND_MAX));
				}
			}
		}
	}

	void normalize(int row){
		double sum = 0;
		for (size_t i = 0; i < m; i++) {
	      	sum += mat[row][i];
	    }
	    for(int i = 0; i < m; i++){
	    	mat[row][i] /= sum;
	    }
	}
};

class Vector{
public:
	double* vec;
	int n;

	Vector(int n, bool random){
		this->n = n;
		vec = new double[n];
		if(random){
			for(int i = 0; i < n; i++){
				vec[i] = ((double) rand() / (RAND_MAX));
			}
		}
		else{
			memset(vec, 0, n * sizeof(double));
		}
	}	
};

Matrix multiple(Matrix mat1, Matrix mat2){
	if(mat1.m != mat2.n){
		cout << "Dimension mismatch!!!" << endl;
		return NULL;
	}
	int n, m, l;
	double sum;
	n = mat1.n;
	m = mat1.m;
	l = mat2.m;
	Matrix result(n, l, false);

	for(int i = 0; i < n; i++){
		for(int j = 0; j < l; j++){
			sum = 0;
			for(int k = 0; k < m; k++){
				sum += (mat1[i][k] * mat2[k][j]);
			}
			result[i][j] = sum;
		}
	}
	return result;
}
