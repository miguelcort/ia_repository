"""
Lección: 02-inference-platform-economics
Fase: 17
Inference platform economics: per-token
pricing, batch discounts, reserved
capacity, spot pricing, TCO calculations.
"""
from __future__ import annotations


class InferencePlan:
    def __init__(self, name, per_token_input, per_token_output,
                 batch_discount=0.0, reserved_discount=0.0):
        self.name = name
        self.per_token_input = per_token_input
        self.per_token_output = per_token_output
        self.batch_discount = batch_discount
        self.reserved_discount = reserved_discount

    def cost(self, input_tokens, output_tokens, batch=False, reserved=False):
        c = self.per_token_input * input_tokens + self.per_token_output * output_tokens
        if batch:
            c *= (1 - self.batch_discount)
        if reserved:
            c *= (1 - self.reserved_discount)
        return c


def tco(plan, monthly_input_tokens, monthly_output_tokens,
        batch_ratio=0.0, reserved=False, fixed_monthly=0.0):
    """Total cost of ownership: per-request + fixed."""
    non_batch = monthly_input_tokens * (1 - batch_ratio)
    batch = monthly_input_tokens * batch_ratio
    non_batch_output = monthly_output_tokens * (1 - batch_ratio)
    batch_output = monthly_output_tokens * batch_ratio
    non_batch_cost = plan.cost(non_batch, non_batch_output, batch=False, reserved=reserved)
    batch_cost = plan.cost(batch, batch_output, batch=True, reserved=reserved)
    return non_batch_cost + batch_cost + fixed_monthly


def per_request_cost(plan, input_tokens, output_tokens, **kwargs):
    return plan.cost(input_tokens, output_tokens, **kwargs)


def main() -> int:
    p = InferencePlan("on_demand", 1e-5, 3e-5, batch_discount=0.5, reserved_discount=0.3)
    print(f"100k/10k: ${per_request_cost(p, 100_000, 10_000):.4f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())