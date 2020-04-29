#include "pypet/core/parser.h"

#include "torch/csrc/jit/frontend/lexer.h"

#include <stdexcept>

namespace pypet {

PYBIND11_MODULE(_parser, m) {
  m.def("parse_scop", [](const std::string& src) {
    TorchParser p(src);

    ScopParser scop_parser(p.Parse());
    // scop_parser.DumpAST();
    scop_parser.Parse();
  });

  py::class_<PypetScop>(m, "PypetScop").def(py::init());
}

void ScopParser::Parse() { pImpl->ParseFunction(); }

void ParserImpl::ParseDecl() { std::cout << ast_.name(); }

void ParserImpl::ParseBody() {
  auto stmts_list = ast_.statements();
  for (auto begin = stmts_list.begin(); begin != stmts_list.end(); ++begin) {
    auto stmt = *begin;
    switch (stmt.kind()) {
      case torch::jit::TK_IF:
        emitIf(torch::jit::If(stmt));
        break;
      case torch::jit::TK_WHILE:
        emitWhile(torch::jit::While(stmt));
        break;
      case torch::jit::TK_FOR:
        emitFor(torch::jit::For(stmt));
        break;
      case torch::jit::TK_ASSIGN:
        emitAssignment(torch::jit::Assign(stmt));
        break;
      case torch::jit::TK_AUG_ASSIGN:
        emitAugAssignment(torch::jit::AugAssign(stmt));
        break;
      case torch::jit::TK_EXPR_STMT: {
        auto expr = torch::jit::ExprStmt(stmt).expr();
        emitExpr(expr);
      } break;
      case torch::jit::TK_RAISE:
        emitRaise(torch::jit::Raise(stmt));
        break;
      case torch::jit::TK_ASSERT:
        emitAssert(torch::jit::Assert(stmt));
        break;
      case torch::jit::TK_RETURN:
        emitReturn(torch::jit::Return(stmt));
        break;
      case torch::jit::TK_CONTINUE:
        emitContinue(torch::jit::Continue(stmt));
        break;
      case torch::jit::TK_BREAK:
        emitBreak(torch::jit::Break(stmt));
        break;
      case torch::jit::TK_PASS:
        // Emit nothing for pass
        break;
      case torch::jit::TK_DEF:
        emitClosure(torch::jit::Def(stmt));
        break;
      case torch::jit::TK_DELETE:
        emitDelete(torch::jit::Delete(stmt));
        break;
      default:
        throw std::invalid_argument("Unrecognized statement kind ");
    }
  }
}

void ParserImpl::ParseFunction() {
  ParseDecl();
  ParseBody();
}

void ParserImpl::emitFor(const torch::jit::For& stmt) {
  std::cout << stmt.range() << std::endl;
}

void ParserImpl::emitIf(const torch::jit::If& stmt) {}

void ParserImpl::emitWhile(const torch::jit::While& stmt) {
  throw std::invalid_argument("while statement is not supported.");
}

void ParserImpl::emitAssignment(const torch::jit::Assign& stmt) {}
void ParserImpl::emitAugAssignment(const torch::jit::AugAssign& stmt) {}
void ParserImpl::emitRaise(const torch::jit::Raise& stmt) {}
void ParserImpl::emitAssert(const torch::jit::Assert& stmt) {}
void ParserImpl::emitReturn(const torch::jit::Return& stmt) {}
void ParserImpl::emitContinue(const torch::jit::Continue& stmt) {}
void ParserImpl::emitBreak(const torch::jit::Break& stmt) {}
void ParserImpl::emitClosure(const torch::jit::Def& stmt) {}
void ParserImpl::emitDelete(const torch::jit::Delete& stmt) {}
void ParserImpl::emitExpr(const torch::jit::Expr& tree) {}

}  // namespace pypet
