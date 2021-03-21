using System;
using System.Linq;
using System.Collections.Generic;

namespace NEA.NeuralNetwork
{
    using Tensor;

    /// <summary>
    /// Provides static functional implementations of statistical loss/error functions.
    /// </summary>
    public static class LossFunctions
    {
        // Checks 2 tensors are of equal dimensions
        private static void checkShapeEqual(Tensor A, Tensor B)
        {
            if (Enumerable.SequenceEqual(A.Shape, B.Shape))
            {
                return;
            }
            else
            {
                throw new Exception("Input and target dimensions must be equal.");
            }
        }

        // Checks 2 matricies are of equal dimensions.
        private static void checkShapeEqual(Matrix A, Matrix B)
        {
            if (Enumerable.SequenceEqual(A.Shape, B.Shape))
            {
                return;
            }
            else
            {
                throw new Exception("Input and target dimensions must be equal.");
            }
        }

        /// <summary>
        /// Measures the mean squared error (squared L2 norm) between each element of each batch in the input x and target y.
        /// </summary>
        /// <param name="x">The input.</param>
        /// <param name="y">The target.</param>
        /// <returns>An array showing the the mean squared error (squared L2 norm) of each batch of the input x and target y.</returns>
        public static float[] MSELoss(Tensor x, Tensor y)
        {
            checkShapeEqual(x, y);
            var loss = new float[x.Shape[0]]; ;
            for (int i = 0; i < x.Shape[0]; i++)
            {
                loss[i] = MSELoss(x[i], y[i]);
            }
            return loss;
        }

        /// <summary>
        /// Measures the mean squared error (squared L2 norm) between each element in the input x and target y.
        /// </summary>
        /// <param name="x">The input.</param>
        /// <param name="y">The target.</param>
        /// <returns>A float showing the the mean squared error (squared L2 norm) between each element of the input x and target y.</returns>
        public static float MSELoss(Matrix x, Matrix y)
        {
            checkShapeEqual(x, y);
            float loss = 0; // accumulator for the total loos
            float nItems = x.Shape[0] * x.Shape[1]; // total number of items in the matrix.
            for (int i = 0; i < x.Shape[0]; i++)
            {
                for (int j = 0; j < x.Shape[1]; j++)
                {
                    loss += MathF.Pow(x[i, j] - y[i, j], 2);
                }
            }
            loss /= nItems;
            return loss;
        }

        /// <summary>
        /// Measures the cross entropy loss between each element in the input x and target y.
        /// </summary>
        /// <param name="x">The input, containing the raw, unnormalized scores for each class.</param>
        /// <param name="y">The target, a batch of one-hot vectors.</param>
        /// <returns>A float array showing the cross entropy loss of each batch of the input x and target y.</returns>
        public static float[] CrossEntropyLoss(Tensor x, Tensor y)
        {
            checkShapeEqual(x, y);
            var loss = new float[x.Shape[0]];
            for (int i = 0; i < x.Shape[0]; i++)
            {
                CrossEntropyLoss(x[i], y[i]);
            }
            return loss;
        }

        /// <summary>
        /// Measures the cross entropy loss between each element in the input x and target y.
        /// </summary>
        /// <param name="x">The input, containing the raw, unnormalized scores for each class.</param>
        /// <param name="y">The target, a one-hot vector.</param>
        /// <returns>A float showing the cross entropy loss of the input x and target y.</returns>
        public static float CrossEntropyLoss(Matrix x, Matrix y)
        {
            float loss = 0;
            int targetIdx = 0;
            for (int i = 0; i < x.Shape[0]; i++)
            {
                if (y[i] == 1)
                {
                    targetIdx = i; // finds the index of the correct output node
                }
                else
                {
                    loss += MathF.Exp(x[i]); // finds sum of exponents
                }
            }
            loss = MathF.Log(loss); // takes log of the sum of exponents
            loss -= x[targetIdx]; // subtracts the output for the correct class
            return loss;
        }
    }
}