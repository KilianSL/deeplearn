namespace NEA.Utils.Data
{
    using Tensor;

    /// <summary>
    /// Stores the input and output tensors for a single batch of data.
    /// </summary>
    public struct Batch
    {
        /// <summary>
        /// Creates a new Batch.
        /// </summary>
        /// <param name="input">The input tensor.</param>
        /// <param name="output">The corresponding output tensor.</param>
        public Batch(Tensor input, Tensor output)
        {
            this.input = input;
            this.output = output;
        }

        /// <summary>
        /// A tensor of input samples.
        /// </summary>
        public Tensor input;

        /// <summary>
        /// A tensor of output samples.
        /// </summary>
        public Tensor output;

        /// <summary>
        /// The amount of items in this batch.
        /// </summary>
        public int BatchSize { get => input.Shape[0]; }
    }
}