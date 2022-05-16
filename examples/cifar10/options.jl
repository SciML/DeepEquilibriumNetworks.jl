using ArgParse


# Parse Training Arguments
function parse_commandline_arguments()
    parse_settings = ArgParseSettings("FastDEQ CIFAR-10 Training")

    @add_arg_table! parse_settings begin
        "--model-size"
            default = "TINY"
            range_tester = x -> x ∈ ("TINY", "LARGE")
            help = "model size: `TINY` or `LARGE`"
        "--model-type"
            default = "VANILLA"
            range_tester = x -> x ∈ ("VANILLA", "SKIP", "SKIPV2")
            help = "model type: `VANILLA`, `SKIP` or `SKIPV2`"
        "--eval-batchsize"
            help = "batch size for evaluation (per process)"
            arg_type = Int
            default = 32
        "--train-batchsize"
            help = "batch size for training (per process)"
            arg_type = Int
            default = 32
        "--discrete"
            help = "use discrete DEQ"
            action = :store_true
        "--jfb"
            help = "enable jacobian-free-backpropagation"
            action = :store_true
        "--abstol"
            default = 0.25f0
            arg_type = Float32
            help = "absolute tolerance for termination"
        "--reltol"
            default = 0.25f0
            arg_type = Float32
            help = "relative tolerance for termination"
        "--w-skip"
            default = 1.0f0
            arg_type = Float32
            help = "weight for skip DEQ loss"
        "--start-epoch"
            help = "manual epoch number (useful on restarts)"
            arg_type = Int
            default = 0
        "--print-freq"
            help = "print frequency"
            arg_type = Int
            default = 100
        "--resume"
            help = "resume from checkpoint"
            arg_type = String
            default = ""
        "--evaluate"
            help = "evaluate model on validation set"
            action = :store_true
        "--seed"
            help = "seed for initializing training. "
            arg_type = Int
            default = 0
        "--checkpoint-dir"
            help = "directory to save checkpoints"
            arg_type = String
            default = "checkpoints/"
        "--log-dir"
            help = "directory to save logs"
            arg_type = String
            default = "logs/"
    end

    return parse_args(parse_settings)
end