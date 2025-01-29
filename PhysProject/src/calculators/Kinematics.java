package calculators;

public class Kinematics {
    public static double finalVelocity(double u, double a, double t) {
        return u + (a * t);
    }

    public static double displacement(double u, double a, double t) {
        return (u * t) + (0.5 * a * t * t);
    }
}
