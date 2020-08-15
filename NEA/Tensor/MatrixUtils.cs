using System;

namespace NEA.Tensor
{
    internal static class MatrixUtils
    {
        public static float[] RandomGaussian(float mean, float stdDev) // Returns an array of 2 random gaussian variables generated using the Marsaglia-Polar method
        {
            Random random = new Random();
            float randomFloat() { return (float)random.NextDouble(); } // gets a random float value from U(0,1)
            float s = 5;
            float u1 = 1;
            float u2 = 1;
            while (s > 1)
            {
                u1 = randomFloat();
                u2 = randomFloat();
                u1 = 2 * u1 - 1;
                u2 = 2 * u2 - 2;
                s = MathF.Pow(u1, 2) + MathF.Pow(u2, 2);
            }
            u1 = MathF.Sqrt((-2 * MathF.Log(s)) / s) * u1;
            u2 = MathF.Sqrt((-2 * MathF.Log(s)) / s) * u2;
            return new float[] { u1 * stdDev + mean, u2 * stdDev + mean };
        }
    }
}