package calculators;

import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        int choice;

        do {
            System.out.println("==== PHYSICS CALCULATOR ====");
            System.out.println("1. Kinematics");
            System.out.println("2. Newton’s Laws (Forces)");
            System.out.println("3. Energy & Work");
            System.out.println("4. Momentum");
            System.out.println("5. Exit");
            System.out.print("Select a topic: ");
            
            choice = scanner.nextInt();

            switch (choice) {
                case 1:
                    kinematicsMenu(scanner);
                    break;
                case 2:
                    forcesMenu(scanner);
                    break;
                case 3:
                    energyMenu(scanner);
                    break;
                case 4:
                    momentumMenu(scanner);
                    break;
                case 5:
                    System.out.println("Exiting program. Goodbye!");
                    break;
                default:
                    System.out.println("Invalid choice. Please try again.");
            }
            System.out.println();
        } while (choice != 5);

        scanner.close();
    }

    private static void kinematicsMenu(Scanner scanner) {
        System.out.println("\n-- Kinematics Calculator --");
        System.out.println("1. Final Velocity (v = u + at)");
        System.out.println("2. Displacement (s = ut + 0.5at²)");
        System.out.print("Choose an equation: ");
        int kChoice = scanner.nextInt();

        if (kChoice == 1) {
            System.out.print("Enter initial velocity (u) in m/s: ");
            double u = scanner.nextDouble();
            System.out.print("Enter acceleration (a) in m/s²: ");
            double a = scanner.nextDouble();
            System.out.print("Enter time (t) in seconds: ");
            double t = scanner.nextDouble();

            double v = Kinematics.finalVelocity(u, a, t);
            System.out.println("Final Velocity: " + v + " m/s");
        } else if (kChoice == 2) {
            System.out.print("Enter initial velocity (u) in m/s: ");
            double u = scanner.nextDouble();
            System.out.print("Enter acceleration (a) in m/s²: ");
            double a = scanner.nextDouble();
            System.out.print("Enter time (t) in seconds: ");
            double t = scanner.nextDouble();

            double s = Kinematics.displacement(u, a, t);
            System.out.println("Displacement: " + s + " meters");
        } else {
            System.out.println("Invalid choice.");
        }
    }

    private static void forcesMenu(Scanner scanner) {
        System.out.println("\n-- Newton’s Laws Calculator --");
        System.out.println("1. Force (F = ma)");
        System.out.print("Choose an equation: ");
        int fChoice = scanner.nextInt();

        if (fChoice == 1) {
            System.out.print("Enter mass (m) in kg: ");
            double m = scanner.nextDouble();
            System.out.print("Enter acceleration (a) in m/s²: ");
            double a = scanner.nextDouble();

            double F = Forces.calculateForce(m, a);
            System.out.println("Force: " + F + " N");
        } else {
            System.out.println("Invalid choice.");
        }
    }

    private static void energyMenu(Scanner scanner) {
        System.out.println("\n-- Energy & Work Calculator --");
        System.out.println("1. Kinetic Energy (KE = 0.5 * m * v²)");
        System.out.print("Choose an equation: ");
        int eChoice = scanner.nextInt();

        if (eChoice == 1) {
            System.out.print("Enter mass (m) in kg: ");
            double m = scanner.nextDouble();
            System.out.print("Enter velocity (v) in m/s: ");
            double v = scanner.nextDouble();

            double KE = Energy.kineticEnergy(m, v);
            System.out.println("Kinetic Energy: " + KE + " J");
        } else {
            System.out.println("Invalid choice.");
        }
    }

    private static void momentumMenu(Scanner scanner) {
        System.out.println("\n-- Momentum Calculator --");
        System.out.println("1. Momentum (p = mv)");
        System.out.print("Choose an equation: ");
        int mChoice = scanner.nextInt();

        if (mChoice == 1) {
            System.out.print("Enter mass (m) in kg: ");
            double m = scanner.nextDouble();
            System.out.print("Enter velocity (v) in m/s: ");
            double v = scanner.nextDouble();

            double p = Momentum.calculateMomentum(m, v);
            System.out.println("Momentum: " + p + " kg·m/s");
        } else {
            System.out.println("Invalid choice.");
        }
    }
}

