namespace NEA.NeuralNetwork
{
    /// <summary>
    /// Interface implemented by all neural network modules.
    /// </summary>
    public interface IModule
    {
        /// <summary>
        /// The function to be called on a forward pass.
        /// </summary>
        /// <param name="input">The input sample.</param>
        /// <returns>The result of the forward pass on the input sample.</returns>
        public Tensor.Tensor Forward(Tensor.Tensor input);
    }
}