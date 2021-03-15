namespace NEA.NeuralNetwork.Layers
{
    using Tensor;
    /// <summary>
    /// A placeholder identity operator.
    /// </summary>
    class Identity:Layer
    {
        /// <summary>
        /// Creates a new identity layer.
        /// </summary>
        /// <param name="InputSize">The size of the input sample. This will also be the size of the output sample.</param>
        public Identity(int InputSize)
        {
            this.InputSize = InputSize;
            this.OutputSize = InputSize;
        }
       /// <summary>
       /// A placeholder identity operator.
       /// </summary>
       /// <param name="x">The input sample.</param>
       /// <returns>The input sample.</returns>
        public override Tensor Forward(Tensor x)
        {
            return x;
        }
    }
}
