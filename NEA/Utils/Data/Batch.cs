namespace NEA.Utils.Data
{
    using Tensor;
    public struct Batch
    {
        public Batch(Tensor input, Tensor output)
        {
            this.input = input;
            this.output = output;
        }
        public Tensor input;
        public Tensor output;
    }
}