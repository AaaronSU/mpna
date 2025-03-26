build:
	mpicc hypre_project.c -L/home/aaron/Downloads/hypre-2.32.0/src/hypre/lib -lHYPRE -lm -o hypre_project

run:
	./hypre_project config.txt

# Il faut remplacer -L/home/aaron/Downloads/hypre-2.32.0/src/hypre/lib par le chemin d'installation hypre dédié

# Un exemple du résultat

# N=10

# ##### Schéma implicite linéarisé #####
# Le pas du temps vaut 3.07072e-05
# Le schéma implicite linéarisé a atteint 10000 itérations sans converger et le temps écoulé vaut 0.244375 s
# Solution finale du schéma linéarisé :
#    i=0, x=0.000, u=1.282690
#    i=3, x=0.300, u=1.033960
#    i=6, x=0.600, u=1.000014
#    i=9, x=0.900, u=1.000000

# ##### Schéma Newton-Raphson #####
# La méthode converge à l'itération 9 avec une erreur de 8.263213e-11 et le temps écoulé vaut 0.000979818 s
# Solution finale de Newton :
#    i=0, x=0.000, u=1.799083
#    i=3, x=0.300, u=1.347469
#    i=6, x=0.600, u=1.051140
#    i=9, x=0.900, u=1.005657



