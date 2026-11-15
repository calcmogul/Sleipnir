// Copyright (c) Sleipnir contributors

//! Procedural macros for the `sleipnir` crate. Use them through their
//! re-exports in `sleipnir`.

use proc_macro2::{Spacing, Span, TokenStream, TokenTree};
use quote::{format_ident, quote, quote_spanned};
use syn::spanned::Spanned;

/// Builds constraints from comparisons.
///
/// See `sleipnir::constrain!` for documentation.
#[proc_macro]
pub fn constrain(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    expand(input.into())
        .unwrap_or_else(syn::Error::into_compile_error)
        .into()
}

#[derive(Clone, Copy, PartialEq)]
enum Relation {
    Equal,
    LessEqual,
    GreaterEqual,
}

#[derive(Clone, Copy)]
struct Operator {
    relation: Relation,
    span: Span,
}

impl Operator {
    fn method(&self) -> syn::Ident {
        let name = match self.relation {
            Relation::Equal => "equals",
            Relation::LessEqual => "leq",
            Relation::GreaterEqual => "geq",
        };
        syn::Ident::new(name, self.span)
    }
}

/// Multi-character Rust operators, longest first. Splitting a run of joint
/// punctuation into these keeps `x<=-1.0` as `x <= -1.0` and keeps `>>=`
/// from being read as `> >=`.
const MULTI_CHAR_OPERATORS: [&str; 24] = [
    "<<=", ">>=", "...", "..=", "==", "!=", "<=", ">=", "&&", "||", "+=", "-=", "*=", "/=", "%=",
    "^=", "&=", "|=", "<<", ">>", "..", "::", "->", "=>",
];

fn expand(input: TokenStream) -> syn::Result<TokenStream> {
    let (segments, operators) = split(input)?;

    let operands = segments
        .into_iter()
        .enumerate()
        .map(|(i, segment)| {
            if segment.is_empty() {
                // Report next to the operator this operand belongs to. With no
                // operators, the input is empty and handled below.
                let Some(op) = operators.get(i.saturating_sub(1)) else {
                    return Ok(None);
                };
                return Err(syn::Error::new(op.span, "expected an expression"));
            }
            let expr: syn::Expr = syn::parse2(segment)?;
            if let syn::Expr::Binary(binary) = &expr
                && matches!(binary.op, syn::BinOp::Lt(_) | syn::BinOp::Gt(_))
            {
                return Err(syn::Error::new(
                    binary.op.span(),
                    "strict inequalities aren't supported; use `<=` or `>=`",
                ));
            }
            Ok(Some(expr))
        })
        .collect::<syn::Result<Option<Vec<_>>>>()?
        .unwrap_or_default();

    if operators.is_empty() {
        let message = "expected a constraint: `lhs == rhs`, `lhs <= rhs`, or `lhs >= rhs`";
        return Err(match operands.first() {
            Some(syn::Expr::Assign(assign)) => syn::Error::new(
                assign.eq_token.span(),
                format!("{message}; use `==` for equality"),
            ),
            Some(expr) => syn::Error::new(expr.span(), message),
            None => syn::Error::new(Span::call_site(), message),
        });
    }

    if operators.len() == 1 {
        let op = operators[0];
        let method = op.method();
        let lhs = reference(&operands[0]);
        let rhs = reference(&operands[1]);
        return Ok(quote_spanned! {op.span=>
            ::sleipnir::Compare::#method(#lhs, #rhs)
        });
    }

    // Chained inequality, e.g., `lower <= x <= upper`
    if let Some(op) = operators.iter().find(|op| op.relation == Relation::Equal) {
        return Err(syn::Error::new(op.span, "`==` can't be chained"));
    }
    if let Some(op) = operators
        .iter()
        .find(|op| op.relation != operators[0].relation)
    {
        return Err(syn::Error::new(
            op.span,
            "chained constraints must all use `<=` or all use `>=`",
        ));
    }

    // Bind each operand once so the middle ones aren't evaluated twice
    let names: Vec<_> = (0..operands.len())
        .map(|i| format_ident!("__sleipnir_operand_{}", i, span = Span::mixed_site()))
        .collect();
    let bindings = names.iter().zip(&operands).map(|(name, operand)| {
        let operand = reference(operand);
        quote! { let #name = #operand; }
    });
    let parts = operators.iter().enumerate().map(|(i, op)| {
        let method = op.method();
        let (lhs, rhs) = (&names[i], &names[i + 1]);
        quote_spanned! {op.span=> ::sleipnir::Compare::#method(#lhs, #rhs) }
    });
    Ok(quote! {
        {
            #(#bindings)*
            ::sleipnir::__private::chain([#(#parts),*])
        }
    })
}

/// Returns `&(expr)` with the operand's span, so type errors point at the
/// operand.
fn reference(expr: &syn::Expr) -> TokenStream {
    quote_spanned! {expr.span()=> &(#expr) }
}

/// Splits the input at top-level `==`, `<=`, and `>=` operators. Delimited
/// groups are single token trees, so comparisons inside parentheses, brackets,
/// or braces aren't split.
fn split(input: TokenStream) -> syn::Result<(Vec<TokenStream>, Vec<Operator>)> {
    let tokens: Vec<TokenTree> = input.into_iter().collect();
    let mut segments = vec![TokenStream::new()];
    let mut operators = Vec::new();

    let mut i = 0;
    while i < tokens.len() {
        let TokenTree::Punct(_) = &tokens[i] else {
            segments.last_mut().unwrap().extend([tokens[i].clone()]);
            i += 1;
            continue;
        };

        // Collect the run of joint punctuation starting here
        let mut run = Vec::new();
        while let Some(TokenTree::Punct(punct)) = tokens.get(i) {
            run.push(punct.clone());
            i += 1;
            if punct.spacing() == Spacing::Alone {
                break;
            }
        }

        // Split the run into operators
        let mut j = 0;
        while j < run.len() {
            let rest: String = run[j..].iter().map(|punct| punct.as_char()).collect();
            let len = MULTI_CHAR_OPERATORS
                .iter()
                .find(|op| rest.starts_with(*op))
                .map_or(1, |op| op.len());
            let op = &rest[..len];
            let span = run[j]
                .span()
                .join(run[j + len - 1].span())
                .unwrap_or_else(|| run[j].span());

            let relation = match op {
                "==" => Some(Relation::Equal),
                "<=" => Some(Relation::LessEqual),
                ">=" => Some(Relation::GreaterEqual),
                "!=" => {
                    return Err(syn::Error::new(
                        span,
                        "`!=` isn't a constraint; use `==`, `<=`, or `>=`",
                    ));
                }
                _ => None,
            };
            match relation {
                Some(relation) => {
                    operators.push(Operator { relation, span });
                    segments.push(TokenStream::new());
                }
                None => {
                    let segment = segments.last_mut().unwrap();
                    for punct in &run[j..j + len] {
                        segment.extend([TokenTree::Punct(punct.clone())]);
                    }
                }
            }
            j += len;
        }
    }

    Ok((segments, operators))
}
