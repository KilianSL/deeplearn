using System;
using System.Linq;
namespace NEA.NeuralNetwork.ActivationFunctions
{
    using Tensor;
    /// <summary>
    /// Provides static functional implementations of neural network activation functions. 
    /// </summary>
    static class ActivationFunctions
    {
        /// <summary>
        /// Applies the element-wise sigmoid function.
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns>The input tensor, with the sigmoid function applied to each value.</returns>
        /// 
        public static Tensor Sigmoid(Tensor x)
        {
            for (int i = 0; i < x.Shape[0]; i++)
            {
                x[i] = Sigmoid(x[i]);
            }
            return x;
        }
        /// <summary>
        /// Applies the element-wise sigmoid function.
        /// </summary>
        /// <param name="x">The input matrix.</param>
        /// <returns>The input matrix, with the sigmoid function applied to each value.</returns>
        /// 
        public static Matrix Sigmoid(Matrix x)
        {
            for (int i = 0; i < x.Shape[0]; i++)
            {
                for (int j = 0; j < x.Shape[1]; j++)
                {
                    x[i, j] = 1 / (1 + MathF.Exp(-x[i,j]));
                }
            }
            return x;
        }
        /// <summary>
        /// Applies the element-wise rectified linear unit function.
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns>The input tensor, with the rectified linear unit function applied to each value.</returns>
        /// 
        public static Tensor ReLU(Tensor x)
        {
            for (int i = 0; i < x.Shape[0]; i++)
            {
                x[i] = ReLU(x[i]);
            }
            return x;
        }
        /// <summary>
        /// Applies the element-wise rectified linear unit function.
        /// </summary>
        /// <param name="x">The input matrix.</param>
        /// <returns>The input matrix, with the rectified linear unit function applied to each value.</returns>
        /// 
        public static Matrix ReLU(Matrix x)
        {
            for (int i = 0; i < x.Shape[0]; i++)
            {
                for (int j = 0; j < x.Shape[1]; j++)
                {
                    x[i, j] = MathF.Max(0, x[i, j]);
                }
            }
            return x;
        }
        /// <summary>
        /// Applies the element-wise hyperbolic tangent function.
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns>The input tensor, with the hyperbolic tangent function applied to each value.</returns>
        /// 
        public static Tensor Tanh(Tensor x)
        {
            for (int i = 0; i < x.Shape[0]; i++)
            {
                x[i] = Tanh(x[i]);
            }
            return x;
        }
        /// <summary>
        /// Applies the element-wise hyperbolic tangent function.
        /// </summary>
        /// <param name="x">The input matrix.</param>
        /// <returns>The input matrix, with the hyperbolic tangent function applied to each value.</returns>
        /// 
        public static Matrix Tanh(Matrix x)
        {
            for (int i = 0; i < x.Shape[0]; i++)
            {
                for (int j = 0; j < x.Shape[1]; j++)
                {
                    x[i, j] = MathF.Tanh(x[i, j]);
                }
            }
            return x;
        }
        /// <summary>
        /// Applies the Softmax function to each batch of an n-dimensional input tensor rescaling them so that the elements of the n-dimensional output tensor lie in the range [0,1] and sum to 1.
        /// </summary>
        /// <param name="x">The input tensor.</param>
        /// <returns>The input tensor, with the softmax function applied to each batch.</returns>
        /// 
        public static Tensor Softmax(Tensor x)
        {
            for (int i = 0; i < x.Shape[0]; i++)
            {
                x[i] = Softmax(x[i]);
            }
            return x;
        }
        /// <summary>
        /// Applies the Softmax function to an input matrix, rescaling each element so that the elements of the matrix lie in the range [0,1] and sum to 1.
        /// </summary>
        /// <param name="x">The input matrix.</param>
        /// <returns>The input matrix, with the softmax function applied.</returns>
        public static Matrix Softmax(Matrix x)
        {
            // Casts the matrix to an IEnumerable, then applies the exponent to each element, then takes the sum of the result.
            float sum = x.Cast<float>().Select(i => MathF.Exp(i)).Sum();
            for (int i = 0; i < x.Shape[0]; i++)
            {
                for (int j = 0; j < x.Shape[1]; j++)
                {
                    x[i, j] = MathF.Exp(x[i, j]) / sum;
                }
            }
            return x;
        }

    }
}
