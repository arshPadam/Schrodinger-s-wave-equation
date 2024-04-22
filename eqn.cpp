/* eqn.cpp: Solution of the one-dimensional Schrodinger equation for
a particle in a harmonic potential, using the shooting method.
To compile and link with gnu compiler, type: g++ -o eqn eqn.cpp
To run the current C++ program, simply type: osci
Plot by gnuplot: /GNUPLOT> set terminal windows
/GNUPLOT> plot "psi-osc.dat" with lines */

#include <cstdio>
#include <cstdlib>
#include <cmath>

#define MAX(a, b) (((a) > (b)) ? (a) : (b))


int main(int argc, char *argv[])
{                                        // Runtime constants
    const static double Epsilon = 1e-10; // Defines the precision of
    //... energy calculations
    const static int N_of_Divisions = 1000;
    const static int N_max = 5; // Number of calculated Eigenstates
    FILE *Wavefunction_file, *Energy_file, *Potential_file;
    Wavefunction_file = fopen("psi-osc.dat", "w");
    Energy_file = fopen("E_n_Oszillator.dat", "w");
    Potential_file = fopen("HarmonicPotentialNoDim.dat", "w");

    if (!(Wavefunction_file && Energy_file && Potential_file))
    {
        printf("Problems to create files output.\n");
        exit(2);
    }

    /* Physical parameters using dimensionless quantities.
    ATTENTION: We set initially: hbar = m = omega = a = 1, and
    reintroduce physical values at the end. According to Eq.(4.117),
    the ground state energy then is E_n = 0.5. Since the wave function
    vanishes only at -infinity and +infinity, we have to cut off the
    calculation somewhere, as given by ’xRange’. If xRange is chosen
    too large, the open (positive) end of the wave function can
    diverge numerically in this simple shooting approach. */


    const static double xRange = 12; // xRange=11.834 corresponds to a
    //... physical range of -20fm < x < +20fm, see after Eq.(4.199).
    const static double h_0 = xRange / N_of_Divisions;
    double *E_pot = new double[N_of_Divisions + 1];
    double dist;

    for (int i = 0; i <= N_of_Divisions; ++i)
    { // Harmonic potential, as given in Eq. (4.115), but dimensionless
        dist = i * h_0 - 0.5 * xRange;
        E_pot[i] = 0.5 * dist * dist; // E_pot[i]=0;//E_pot=0:Infinite Well!
        fprintf(Potential_file, "%16.12e \t\t %16.12e\n", dist, E_pot[i]);
    }
    fclose(Potential_file);

    /* Since the Schrodinger equation is linear, the amplitude of the
    wavefunction will be fixed by normalization.
    At left we set it small but nonzero. */

    const static double Psi_left = 1.0e-3; // left boundary condition
    const static double Psi_right = 0.0;   // right boundary condition
    double *Psi, *EigenEnergies;           // Arrays to hold the results
    Psi = new double[N_of_Divisions + 1];  // N_of_Points = N_of_Divisions+1
    EigenEnergies = new double[N_max + 1];
    Psi[0] = Psi_left;
    Psi[1] = Psi_left + 1.0e-3; // Add arbitrary small value
    int N_quantum;              // N_quantum is Energy Quantum Number
    int Nodes_plus;             // Number of nodes (+1) in wavefunction
    double K_square;            // Square of wave vector

    // Initial Eigen-energy search limits

    double E_lowerLimit = 0.0; // Eigen-energy must be positive
    double E_upperLimit = 10.0;
    int End_sign = -1;
    bool Limits_are_defined = false;
    double Normalization_coefficient;
    double E_trial;
    // MAIN LOOP begins:-----------------------------------

    for (N_quantum = 1; N_quantum <= N_max; ++N_quantum)
    {
        // Find the eigen-values for energy. See theorems (4.1) and (4.2).
        Limits_are_defined = false;
        while (Limits_are_defined == false)
        { /* First, determine an upper limit for energy, so that the wavefunction
        Psi[i] has one node more than physically needed. */
            Nodes_plus = 0;
            E_trial = E_upperLimit;
            for (int i = 2; i <= N_of_Divisions; ++i)
            {
                K_square = 2.0 * (E_trial - E_pot[i]);
                // Now use the NUMEROV-equation (4.197) to calculate wavefunction
                Psi[i] = 2.0 * Psi[i - 1] * (1.0 - (5.0 * h_0 * h_0 * K_square / 12.0)) / (1.0 + (h_0 * h_0 * K_square / 12.0)) - Psi[i - 2];
                if (Psi[i] * Psi[i - 1] < 0)
                    ++Nodes_plus;
            }
            /* If one runs into the following condition, the modification
            of the upper limit was too aggressive. */
            if (E_upperLimit < E_lowerLimit)
                E_upperLimit = MAX(2 * E_upperLimit, -2 * E_upperLimit);
            if (Nodes_plus > N_quantum)
                E_upperLimit *= 0.7;
            else if (Nodes_plus < N_quantum)
                E_upperLimit *= 2.0;
            else
                Limits_are_defined = true; // At least one node should appear.
        }                                  // End of the loop: while (Limits_are_defined == false)
        // Refine the energy by satisfying the right boundary condition.
        End_sign = -End_sign;
        while ((E_upperLimit - E_lowerLimit) > Epsilon)
        {
            E_trial = (E_upperLimit + E_lowerLimit) / 2.0;
            for (int i = 2; i <= N_of_Divisions; ++i)
            { // Again eq.(4.197) of the Numerov-algorithm:
                K_square = 2.0 * (E_trial - E_pot[i]);
                Psi[i] = 2.0 * Psi[i - 1] * (1.0 - (5.0 * h_0 * h_0 * K_square / 12.0)) / (1.0 + (h_0 * h_0 * K_square / 12.0)) - Psi[i - 2];
            }
            if (End_sign * Psi[N_of_Divisions] > Psi_right)
                E_lowerLimit = E_trial;
            else
                E_upperLimit = E_trial;
        } // End of loop: while ((E_upperLimit - E_lowerLimit) > Epsilon)
        // Initialization for the next iteration in main loop

        E_trial = (E_upperLimit + E_lowerLimit) / 2;
        EigenEnergies[N_quantum] = E_trial;
        E_upperLimit = E_trial;
        E_lowerLimit = E_trial;
        // Now find the normalization coefficient

        double Integral = 0.0;
        for (int i = 1; i <= N_of_Divisions; ++i)
        { // Simple integration
            Integral += 0.5 * h_0 * (Psi[i - 1] * Psi[i - 1] + Psi[i] * Psi[i]);
        }
        Normalization_coefficient = sqrt(1.0 / Integral);
        // Output of normalized dimensionless wave function

        for (int i = 0; i <= N_of_Divisions; ++i)
        {
            fprintf(Wavefunction_file, "%16.12e \t\t %16.12e\n",
                    i * h_0 - 0.5 * xRange, Normalization_coefficient * Psi[i]);
        }
        fprintf(Wavefunction_file, "\n");
    } // End of MAIN LOOP. --------------------------------
    fclose(Wavefunction_file);


    /*Finally convert dimensionless units in real units. Note that
    energy does not depend explicitly on the particle’s mass anymore:
    hbar = 1.05457e-34;// Planck constant/2pi
    omega = 5.34e21; // Frequency in 1/s
    MeV = 1.602176487e-13; // in J
    The correct normalization would be hbar*omega/MeV = 3.5148461144,
    but we use the approximation 3.5 for energy-scale as in chap. 4.9 */
    const static double Energyscale = 3.5; // in MeV
    // Output with rescaled dimensions; assign Energy_file

    printf("Quantum Harmonic Oscillator, program osci.cpp\n");
    printf("Energies in MeV:\n");
    printf("n \t\t E_n\n");
    
    for (N_quantum = 1; N_quantum <= N_max; ++N_quantum)
    {
        fprintf(Energy_file, "%d \t\t %16.12e\n", N_quantum - 1,
                Energyscale * EigenEnergies[N_quantum]);
        printf("%d \t\t %16.12e\n", N_quantum - 1,
               Energyscale * EigenEnergies[N_quantum]);
    }
   
    fprintf(Energy_file, "\n");
    fclose(Energy_file);
    printf("Wave-Functions in File: psi_osc.dat \n");
    printf("\n");
    return 0;
}