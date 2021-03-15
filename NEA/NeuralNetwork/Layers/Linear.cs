using System;
namespace NEA.NeuralNetwork.Layers
{
    using Tensor;
    /// <summary>
    /// Applies a linear transformation to the incoming data. 
    /// </summary>
    class Linear:Layer
    {
        // Class attributes
        private Matrix weight; // The weights for this layer
        private Matrix bias; // The bias for this layer

        // Constructor
        /// <summary>
        /// Creates a new linear layer.
        /// </summary>
        /// <param name="InputSize">The size of the input sample.</param>
        /// <param name="OutputSize">This size of the output sample.</param>
        public Linear(int InputSize, int OutputSize)
        {
            this.InputSize = InputSize;
            this.OutputSize = OutputSize;
            ResetParameters();
        }

        /// <summary>
        /// Applies a linear transformation to the incoming data. 
        /// </summary>
        /// <param name="x">The input data.</param>
        /// <returns>The transformed input data.</returns>
        public override Tensor Forward(Tensor x)
        {
            return new Tensor(1,1,1);
        }

        /// <summary>
        /// Randomly initialises the weights and biases for this layer. 
        /// </summary>
        public void ResetParameters()
        {
            this.weight = Matrix.GaussianMatrix(OutputSize, InputSize, stdDev: 1.0f / MathF.Sqrt(InputSize));
            this.bias = Matrix.ZeroMatrix(OutputSize, 1);
        }
    }
}

