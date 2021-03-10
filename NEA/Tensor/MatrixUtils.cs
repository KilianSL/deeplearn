using System;

namespace NEA.Tensor
{
    internal static class MatrixUtils
    {
        /// <summary>
        /// Draws variables from a random Gaussian distribution.
        /// </summary>
        /// <param name="mean">The mean of the distribution.</param>
        /// <param name="stdDev">The standard deviation of the distribution.</param>
        /// <returns>An array of 2 random variables drawn from the distribution.</returns>
        public static float[] RandomGaussian(float mean, float stdDev) 
        {
            Random random = new Random();
            // gets a random float value from U(0,1)
            float randomFloat() { return (float)random.NextDouble(); } 
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