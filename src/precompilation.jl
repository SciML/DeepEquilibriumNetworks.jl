using PrecompileTools: @compile_workload, @setup_workload

# Precompilation workload is skipped as it often fails with AD
@setup_workload begin
    @compile_workload begin
    end
end
