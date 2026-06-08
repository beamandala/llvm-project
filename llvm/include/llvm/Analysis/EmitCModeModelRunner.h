#ifndef LLVM_ANALYSIS_EMITCMODEMODELRUNNER_H
#define LLVM_ANALYSIS_EMITCMODEMODELRUNNER_H

#include "llvm/Analysis/MLModelRunner.h"
#include "llvm/Analysis/TensorSpec.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include "output.h"

#include <cassert>

namespace llvm {

class EmitCModeModelRunner final : public MLModelRunner {
public:
  EmitCModeModelRunner(LLVMContext &Ctx, const std::vector<TensorSpec> &InputSpec,
                       StringRef DecisionName)
      : MLModelRunner(Ctx, MLModelRunner::Kind::Release, InputSpec.size()),
        CompiledModel(std::make_unique<actionClass>()),
        NumInputs(InputSpec.size()) {
    for (size_t I = 0; I < InputSpec.size(); ++I) {
      setUpBufferForTensor(I, InputSpec[I], nullptr);
      void *Buffer = getTensorUntyped(I);
      StringRef Name = InputSpec[I].name();
      if (Name == "callsite_cost") callsite_cost_ptr = (int64_t*)Buffer;
      else if (Name == "is_multiple_blocks") is_multiple_blocks_ptr = (int64_t*)Buffer;
      else if (Name == "caller_conditionally_executed_blocks") caller_conditionally_executed_blocks_ptr = (int64_t*)Buffer;
      else if (Name == "inlining_default") inlining_default_ptr = (int64_t*)Buffer;
      else if (Name == "cold_cc_penalty") cold_cc_penalty_ptr = (int64_t*)Buffer;
      else if (Name == "callee_conditionally_executed_blocks") callee_conditionally_executed_blocks_ptr = (int64_t*)Buffer;
      else if (Name == "callee_users") callee_users_ptr = (int64_t*)Buffer;
      else if (Name == "callee_basic_block_count") callee_basic_block_count_ptr = (int64_t*)Buffer;
      else if (Name == "nr_ctant_params") nr_ctant_params_ptr = (int64_t*)Buffer;
      else if (Name == "load_relative_intrinsic") load_relative_intrinsic_ptr = (int64_t*)Buffer;
      else if (Name == "jump_table_penalty") jump_table_penalty_ptr = (int64_t*)Buffer;
      else if (Name == "unsimplified_common_instructions") unsimplified_common_instructions_ptr = (int64_t*)Buffer;
      else if (Name == "indirect_call_penalty") indirect_call_penalty_ptr = (int64_t*)Buffer;
      else if (Name == "load_elimination") load_elimination_ptr = (int64_t*)Buffer;
      else if (Name == "call_penalty") call_penalty_ptr = (int64_t*)Buffer;
      else if (Name == "cost_estimate") cost_estimate_ptr = (int64_t*)Buffer;
      else if (Name == "case_cluster_penalty") case_cluster_penalty_ptr = (int64_t*)Buffer;
      else if (Name == "node_count") node_count_ptr = (int64_t*)Buffer;
      else if (Name == "call_argument_setup") call_argument_setup_ptr = (int64_t*)Buffer;
      else if (Name == "sroa_savings") sroa_savings_ptr = (int64_t*)Buffer;
      else if (Name == "lowered_call_arg_setup") lowered_call_arg_setup_ptr = (int64_t*)Buffer;
      else if (Name == "threshold") threshold_ptr = (int64_t*)Buffer;
      else if (Name == "dead_blocks") dead_blocks_ptr = (int64_t*)Buffer;
      else if (Name == "constant_args") constant_args_ptr = (int64_t*)Buffer;
      else if (Name == "sroa_losses") sroa_losses_ptr = (int64_t*)Buffer;
      else if (Name == "simplified_instructions") simplified_instructions_ptr = (int64_t*)Buffer;
      else if (Name == "num_loops") num_loops_ptr = (int64_t*)Buffer;
      else if (Name == "step_type") step_type_ptr = (int32_t*)Buffer;
      else if (Name == "edge_count") edge_count_ptr = (int64_t*)Buffer;
      else if (Name == "nested_inlines") nested_inlines_ptr = (int64_t*)Buffer;
      else if (Name == "caller_basic_block_count") caller_basic_block_count_ptr = (int64_t*)Buffer;
      else if (Name == "last_call_to_static_bonus") last_call_to_static_bonus_ptr = (int64_t*)Buffer;
      else if (Name == "nested_inline_cost_estimate") nested_inline_cost_estimate_ptr = (int64_t*)Buffer;
      else if (Name == "callsite_height") callsite_height_ptr = (int64_t*)Buffer;
      else if (Name == "constant_offset_ptr_args") constant_offset_ptr_args_ptr = (int64_t*)Buffer;
      else if (Name == "switch_penalty") switch_penalty_ptr = (int64_t*)Buffer;
      else if (Name == "discount") discount_ptr = (float*)Buffer;
      else if (Name == "caller_users") caller_users_ptr = (int64_t*)Buffer;
      else if (Name == "reward") reward_ptr = (float*)Buffer;
    }
  }

  static bool classof(const MLModelRunner *R) {
    return R->getKind() == MLModelRunner::Kind::Release; 
  }

  void *evaluateUntyped() override {
#define COPY_TENSOR(PTR, FIELD)                                                \
  if (PTR)                                                                     \
    std::memcpy(CompiledModel->FIELD, PTR, sizeof(CompiledModel->FIELD));      \
  else                                                                         \
    WithColor::warning() << #PTR " doesn't exist\n";

    COPY_TENSOR(callsite_cost_ptr, fieldName0);
    COPY_TENSOR(is_multiple_blocks_ptr, fieldName1);
    COPY_TENSOR(caller_conditionally_executed_blocks_ptr, fieldName2);
    COPY_TENSOR(inlining_default_ptr, fieldName3);
    COPY_TENSOR(cold_cc_penalty_ptr, fieldName4);
    COPY_TENSOR(callee_conditionally_executed_blocks_ptr, fieldName5);
    COPY_TENSOR(callee_users_ptr, fieldName6);
    COPY_TENSOR(callee_basic_block_count_ptr, fieldName7);
    COPY_TENSOR(nr_ctant_params_ptr, fieldName8);
    COPY_TENSOR(load_relative_intrinsic_ptr, fieldName9);
    COPY_TENSOR(jump_table_penalty_ptr, fieldName10);
    COPY_TENSOR(unsimplified_common_instructions_ptr, fieldName11);
    COPY_TENSOR(indirect_call_penalty_ptr, fieldName12);
    COPY_TENSOR(load_elimination_ptr, fieldName13);
    COPY_TENSOR(call_penalty_ptr, fieldName14);
    COPY_TENSOR(cost_estimate_ptr, fieldName15);
    COPY_TENSOR(case_cluster_penalty_ptr, fieldName16);
    COPY_TENSOR(node_count_ptr, fieldName17);
    COPY_TENSOR(call_argument_setup_ptr, fieldName18);
    COPY_TENSOR(sroa_savings_ptr, fieldName19);
    COPY_TENSOR(lowered_call_arg_setup_ptr, fieldName20);
    COPY_TENSOR(threshold_ptr, fieldName21);
    COPY_TENSOR(dead_blocks_ptr, fieldName22);
    COPY_TENSOR(constant_args_ptr, fieldName23);
    COPY_TENSOR(sroa_losses_ptr, fieldName24);
    COPY_TENSOR(simplified_instructions_ptr, fieldName25);
    COPY_TENSOR(num_loops_ptr, fieldName26);
    COPY_TENSOR(step_type_ptr, fieldName27);
    COPY_TENSOR(edge_count_ptr, fieldName28);
    COPY_TENSOR(nested_inlines_ptr, fieldName29);
    COPY_TENSOR(caller_basic_block_count_ptr, fieldName30);
    COPY_TENSOR(last_call_to_static_bonus_ptr, fieldName31);
    COPY_TENSOR(nested_inline_cost_estimate_ptr, fieldName32);
    COPY_TENSOR(callsite_height_ptr, fieldName33);
    COPY_TENSOR(constant_offset_ptr_args_ptr, fieldName34);
    COPY_TENSOR(switch_penalty_ptr, fieldName35);
    COPY_TENSOR(discount_ptr, fieldName36);
    COPY_TENSOR(caller_users_ptr, fieldName37);
    COPY_TENSOR(reward_ptr, fieldName38);

#undef COPY_TENSOR

    Result = CompiledModel->operator()();
    return &Result;
  }

private:
  std::unique_ptr<actionClass> CompiledModel;
  size_t NumInputs;
  int64_t Result;

  int64_t *callsite_cost_ptr = nullptr;
  int64_t *is_multiple_blocks_ptr = nullptr;
  int64_t *caller_conditionally_executed_blocks_ptr = nullptr;
  int64_t *inlining_default_ptr = nullptr;
  int64_t *cold_cc_penalty_ptr = nullptr;
  int64_t *callee_conditionally_executed_blocks_ptr = nullptr;
  int64_t *callee_users_ptr = nullptr;
  int64_t *callee_basic_block_count_ptr = nullptr;
  int64_t *nr_ctant_params_ptr = nullptr;
  int64_t *load_relative_intrinsic_ptr = nullptr;
  int64_t *jump_table_penalty_ptr = nullptr;
  int64_t *unsimplified_common_instructions_ptr = nullptr;
  int64_t *indirect_call_penalty_ptr = nullptr;
  int64_t *load_elimination_ptr = nullptr;
  int64_t *call_penalty_ptr = nullptr;
  int64_t *cost_estimate_ptr = nullptr;
  int64_t *case_cluster_penalty_ptr = nullptr;
  int64_t *node_count_ptr = nullptr;
  int64_t *call_argument_setup_ptr = nullptr;
  int64_t *sroa_savings_ptr = nullptr;
  int64_t *lowered_call_arg_setup_ptr = nullptr;
  int64_t *threshold_ptr = nullptr;
  int64_t *dead_blocks_ptr = nullptr;
  int64_t *constant_args_ptr = nullptr;
  int64_t *sroa_losses_ptr = nullptr;
  int64_t *simplified_instructions_ptr = nullptr;
  int64_t *num_loops_ptr = nullptr;
  int32_t *step_type_ptr = nullptr;
  int64_t *edge_count_ptr = nullptr;
  int64_t *nested_inlines_ptr = nullptr;
  int64_t *caller_basic_block_count_ptr = nullptr;
  int64_t *last_call_to_static_bonus_ptr = nullptr;
  int64_t *nested_inline_cost_estimate_ptr = nullptr;
  int64_t *callsite_height_ptr = nullptr;
  int64_t *constant_offset_ptr_args_ptr = nullptr;
  int64_t *switch_penalty_ptr = nullptr;
  float *discount_ptr = nullptr;
  int64_t *caller_users_ptr = nullptr;
  float *reward_ptr = nullptr;
};

} // namespace llvm

#endif // LLVM_ANALYSIS_EMITCMODEMODELRUNNER_H