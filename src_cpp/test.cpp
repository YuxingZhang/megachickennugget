#include <iostream>
#include <armadillo>
#include <cmath>
using namespace std;
using namespace arma;

int main(){
	vec h(10, fill::zeros);
	vec g(10, fill::zeros);
	
	h += 3;
	g += 5;
	h(7) = 100;
	cout << h.t() << endl;
	cout << g.t() << endl;
	cout << h / g << endl;
	cout << 1 / h << endl;
	return 0;
}
