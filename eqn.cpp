#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>

using namespace std;

const double hbar = 1.0;  // Planck's constant
const double m = 1.0;     // Mass of the particle
const int N = 1000;       // Number of grid points
const double L = 10.0;    // Length of the domain
const double dx = L / N;  // Spacing between points

// Define the potential function (example: harmonic oscillator)
double potential(double x) {
    return 0.5 * x * x; // Harmonic oscillator potential
}

// Function to perform tridiagonal matrix diagonalization using the QR algorithm
void tridiagonal_qr(vector<double>& diag, vector<double>& off_diag, vector<double>& eigenvalues, vector<vector<double>>& eigenvectors, int max_iter = 1000, double tol = 1e-10) {
    int n = diag.size();

    // Initialize eigenvectors as identity matrix
    eigenvectors.resize(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        eigenvectors[i][i] = 1.0;
    }

    for (int iter = 0; iter < max_iter; ++iter) {
        bool converged = true;

        for (int i = 0; i < n - 1; ++i) {
            if (abs(off_diag[i]) > tol) {
                converged = false;
                double a = diag[i];
                double b = off_diag[i];
                double c = diag[i + 1];
                double tau = (c - a) / (2.0 * b);
                double t = (tau >= 0) ? 1.0 / (tau + sqrt(1.0 + tau * tau)) : 1.0 / (tau - sqrt(1.0 + tau * tau));
                double cs = 1.0 / sqrt(1.0 + t * t);
                double sn = t * cs;

                // Update matrix elements
                diag[i] = a - t * b;
                diag[i + 1] = c + t * b;
                off_diag[i] = 0.0;

                // Rotate off-diagonal elements and eigenvectors
                for (int k = 0; k < n; ++k) {
                    double temp = eigenvectors[k][i];
                    eigenvectors[k][i] = cs * temp - sn * eigenvectors[k][i + 1];
                    eigenvectors[k][i + 1] = sn * temp + cs * eigenvectors[k][i + 1];
                }
            }
        }

        if (converged) break;
    }

    // Copy diagonal elements as eigenvalues
    eigenvalues = diag;
}

int main() {
    // Step 1: Define grid points and potential
    vector<double> x(N);
    for (int i = 0; i < N; ++i) {
        x[i] = -L / 2.0 + i * dx;
    }

    // Step 2: Create the tridiagonal matrix representation of the Hamiltonian
    vector<double> diag(N, 0.0);        // Main diagonal
    vector<double> off_diag(N - 1, 0.0); // Off-diagonal

    for (int i = 0; i < N; ++i) {
        diag[i] = 2.0 / (dx * dx) + potential(x[i]);
        if (i < N - 1) {
            off_diag[i] = -1.0 / (dx * dx);
        }
    }

    // Step 3: Solve for eigenvalues and eigenvectors
    vector<double> eigenvalues;
    vector<vector<double>> eigenvectors;

    tridiagonal_qr(diag, off_diag, eigenvalues, eigenvectors);

    // Step 4: Save results to files
    ofstream eigenvalues_file("eigenvalues.txt");
    ofstream wavefunctions_file("wavefunctions.txt");

    eigenvalues_file << "Eigenvalues:\n";
    for (size_t i = 0; i < eigenvalues.size(); ++i) {
        eigenvalues_file << eigenvalues[i] << endl;
    }

    wavefunctions_file << "Wavefunctions:\n";
    for (int i = 0; i < 5; ++i) { // Save the first 5 wavefunctions
        for (int j = 0; j < N; ++j) {
            wavefunctions_file << x[j] << " " << eigenvectors[j][i] << endl;
        }
        wavefunctions_file << "\n\n";
    }

    eigenvalues_file.close();
    wavefunctions_file.close();

    cout << "Eigenvalues and wavefunctions saved to files.\n";
    return 0;
}
