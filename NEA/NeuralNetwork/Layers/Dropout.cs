using System;

namespace NEA.NeuralNetwork.Layers
{
    using Tensor;

    /// <summary>
    /// During training, randomly zeroes some of the elements of the input tensor with probability p using samples from a uniform distribution.
    /// </summary>
    public class Dropout : Layer
    {
        private readonly float p; // The p value for the dropout function

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

        /// <summary>
        /// During training, randomly zeroes some of the elements of the input tensor with probability p using samples from a uniform distribution.
        /// </summary>
        public override Tensor Forward(Tensor x)
        {
            var rand = new Random();
            for (int i = 0; i < x.Shape[0]; i++) // for each sample
            {
                for (int j = 0; j < x.Shape[1]; j++)
                {
                    for (int k = 0; k < x.Shape[2]; k++)
                    {
                        if (rand.NextDouble() > p)
                        {
                            x[i, j, k] = 0;
                        }
                    }
                }
            }
            return x;
        }
    }
}