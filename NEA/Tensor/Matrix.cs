using System;
using System.Collections.Generic;
using System.Text;

namespace NEA.Tensor
{
    public class Matrix
    {
        public float[,] data;
        private int[] shape;

        // Constructor for empty matrix with specified shape
        public Matrix(int rows, int columns)
        {
            this.data = new float[rows, columns];
            this.shape = new int[] { rows, columns };
        }

        // Constructor for matrix object prefilled with values from a float array
        public Matrix(float[,] data)
        {
            this.data = data;
            this.shape = new int[] { data.GetLength(0), data.GetLength(1) };
        }

        // Higher-order constructor that returns a matrix filled with 0s
        public static Matrix ZeroMatrix(int rows, int columns)
        {
            return new Matrix(rows, columns); // float arrays initialise to 0.0f by default, function simply provided for user ease
        }

        // Higher-order constructor that returns matrix filled with numbers drawn from a guassian distrbution
        public static Matrix GaussianMatrix(int rows, int columns, float mean=0, float stdDev = 1)
        {
            float[,] data = new float[rows, columns];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < columns; j++)
                {
                    data[i, j] = randomGaussian(mean, stdDev)[0];
                }
            }
            return new Matrix(data);
        }

        private static float[] randomGaussian(float mean, float stdDev) // Returns an array of 2 random gaussian variables generated using the Marsaglia-Polar method
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
                u2 = 2 *u2 - 2;
                s = MathF.Pow(u1, 2) + MathF.Pow(u2, 2);
            }
            u1 = MathF.Sqrt((-2 * MathF.Log(s)) / s) * u1;
            u2 = MathF.Sqrt((-2 * MathF.Log(s)) / s) * u2;
            return new float[] { u1 * stdDev + mean, u2 * stdDev + mean };
        }

        public override string ToString()
        {
            return base.ToString();
        }
    }
}
