#include "pypet/core/parser.h"

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
      case torch::jit::TK_IF: {
        EmitIf emitter;
        emitter(torch::jit::If(stmt));
      } break;
      case torch::jit::TK_WHILE: {
        EmitWhile emitter;
        emitter(torch::jit::While(stmt));
      } break;
      case torch::jit::TK_FOR: {
        EmitFor emmiter;
        emmiter(torch::jit::For(stmt));
      } break;
      case torch::jit::TK_ASSIGN: {
        EmitAssignment emitter;
        emitter(torch::jit::Assign(stmt));
      } break;
      case torch::jit::TK_AUG_ASSIGN: {
        EmitAugAssignment emitter;
        emitter(torch::jit::AugAssign(stmt));
      } break;
      case torch::jit::TK_EXPR_STMT: {
        auto expr = torch::jit::ExprStmt(stmt).expr();
        EmitExpr emitter;
        emitter(expr);
      } break;
      case torch::jit::TK_RAISE: {
        EmitRaise emitter;
        emitter(torch::jit::Raise(stmt));
      } break;
      case torch::jit::TK_ASSERT: {
        EmitAssert emitter;
        emitter(torch::jit::Assert(stmt));
      } break;
      case torch::jit::TK_RETURN: {
        EmitReturn emitter;
        emitter(torch::jit::Return(stmt));
      } break;
      case torch::jit::TK_CONTINUE: {
        EmitContinue emitter;
        emitter(torch::jit::Continue(stmt));
      } break;
      case torch::jit::TK_BREAK: {
        EmitBreak emitter;
        emitter(torch::jit::Break(stmt));
      } break;
      case torch::jit::TK_PASS:
        // Emit nothing for pass
        break;
      case torch::jit::TK_DEF: {
        EmitClosure emitter;
        emitter(torch::jit::Def(stmt));
        break;
      }
      case torch::jit::TK_DELETE: {
        EmitDelete emitter;
        emitter(torch::jit::Delete(stmt));
      } break;
      default:
        throw std::invalid_argument("Unrecognized statement kind ");
    }
  }
}

void ParserImpl::ParseFunction() {
  ParseDecl();
  ParseBody();
}

}  // namespace pypet
