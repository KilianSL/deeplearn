using System;

namespace NEA.NeuralNetwork.Layers
{
    using Tensor;

    /// <summary>
    /// During training, randomly zeroes some of the elements of the input tensor with probability p using samples from a uniform distribution.
    /// </summary>
    class Dropout : Layer
    {
        private float p; // The p value for the dropout function

        public Dropout(int InputSize, float p)
        {
            if (0 < p && p < 1)
            {
                throw new Exception("p value must be between 0 and 1");
            }
            this.p = p; ;
            this.InputSize = InputSize;
            this.OutputSize = InputSize;
        }

        public override Tensor Forward(Tensor x)
        {
            return x;
        }
    }
}