import update
import numpy as np

def test_lmd():
	xi = 1
	print "unit testing update.lmd..."
	res = update.lmd(xi)
	truth = 1.0 / (2.0 * xi) * (1.0 / (1.0 + np.exp(-xi)) - 0.5)
	if res == truth:
		print "pass!"
	else:
		print "WRONG!!!"
	print "\n"

def test_update_auxiliary():
	print "unit testing update.update_auxiliary..."
	idx = 0
	Alpha = np.array([0.1])
	Xi = np.array([[1.0,2.0]])
	Var = dict(Sigma = np.array([[1.0,1.0]]), mu = np.array([[0.0,0.0]]))
	Sidx = 2
	Xi_new = np.array([[np.sqrt(1.0 + 0.0 - 2.0 * 0.1 * 0.0 + 0.1 ** 2), np.sqrt(1.0 + 0.0 - 2.0 * 0.1 * 0.0 + 0.1 ** 2)]])
	Alpha_new = np.array([0])
	update.update_auxiliary(idx, Alpha, Xi, Var, Sidx)
	if Xi_new[0][0] == Xi[0][0] and Xi_new[0][1] == Xi[0][1] and Alpha_new[0] == Alpha[0]:
		print "pass!"
	else:
		print "WRONG!!!"
	print "\n"

def test_update_z():
	print "unit testing update.update_z..."
	d = 0
	n = 0
	Z = [np.array([[0.5, 0.5]])]
	Eta = dict(mu = np.array([[0,0]]))
	# Rho = dict(mu = )

def main():
	test_lmd()
	test_update_auxiliary()

if __name__ == "__main__":
    main()
