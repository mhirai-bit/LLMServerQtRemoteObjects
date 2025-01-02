#ifndef LLAMA_RESPONSE_GENERATOR_H
#define LLAMA_RESPONSE_GENERATOR_H

#include "rep_LlamaResponseGenerator_source.h" // 生成されたソースヘッダ
#include "llama.h"
#include <QObject>
#include <QString>

/*
  LlamaResponseGeneratorSimpleSource（または LlamaResponseGeneratorSource）を継承し、
  repファイルで宣言した generate(...) スロットや partialResponseReady(...) シグナルなどを
  実装／emit できるようにする。
*/
class LlamaResponseGenerator : public LlamaResponseGeneratorSimpleSource
{
    Q_OBJECT

public:
    // コンストラクタ: llama_model / llama_context を受け取る
    explicit LlamaResponseGenerator(QObject *parent = nullptr,
                                    llama_model *model = nullptr,
                                    llama_context *ctx = nullptr);

    // デストラクタ: sampler があれば破棄
    ~LlamaResponseGenerator() override;

    // repファイルで定義したスロットをオーバーライドする
    // 「rep_LlamaResponseGenerator_source.h」で
    //   virtual void generate(const QString & request) = 0;
    // と宣言されているため、必須実装
    void generate(const QString &request) override;

private:
    // llama_context / llama_model / llama_sampler
    llama_model   *m_model   { nullptr };
    llama_context *m_ctx     { nullptr };
    llama_sampler *m_sampler { nullptr };

    // 実際にテキスト生成する前の初期化処理
    void initializeSampler();
};

#endif // LLAMA_RESPONSE_GENERATOR_H
