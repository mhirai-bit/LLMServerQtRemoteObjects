#include "llamaresponsegenerator.h"
#include <QDebug>

// Constructor: calls parent class constructor and stores model/context
// コンストラクタ: 親クラスのコンストラクタを呼び出し、model/contextを格納
LlamaResponseGenerator::LlamaResponseGenerator(QObject *parent,
                                               llama_model *model,
                                               llama_context *ctx)
    : LlamaResponseGeneratorSimpleSource(parent) // Call parent (generated) class constructor / 親(生成コード)のコンストラクタ呼び出し
    , m_model(model)
    , m_ctx(ctx)
{
}

// Destructor: frees sampler if allocated
// デストラクタ: samplerが作成されていたら破棄
LlamaResponseGenerator::~LlamaResponseGenerator()
{
    if (m_sampler) {
        llama_sampler_free(m_sampler);
        m_sampler = nullptr;
    }
}

// Overridden generate(...) slot from the .rep file
// repファイルで定義されたgenerate(...)スロットのオーバーライド
void LlamaResponseGenerator::generate(const QString &request)
{
    // If first generation, initialize sampler
    // 初回の生成時にサンプラーを初期化
    if (!m_sampler) {
        initializeSampler();
    }

    // Convert QString to std::string
    // QStringをstd::stringに変換
    std::string response;
    std::string promptStd = request.toStdString();

    // Tokenize prompt text
    // プロンプトテキストをトークン化
    const int n_prompt_tokens = -llama_tokenize(
        m_model,
        promptStd.c_str(),
        promptStd.size(),
        nullptr,
        0,
        true,  // is_prefix
        true   // is_bos
        );

    // Prepare vector for tokenized prompt
    // トークナイズ結果を格納するベクタを用意
    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    if (llama_tokenize(
            m_model,
            promptStd.c_str(),
            promptStd.size(),
            prompt_tokens.data(),
            prompt_tokens.size(),
            llama_get_kv_cache_used_cells(m_ctx) == 0,
            true) < 0)
    {
        Q_EMIT generationError(QStringLiteral("failed to tokenize the prompt"));
        return;
    }

    // Single batch for decoding
    // デコード用に1つのバッチを作成
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    llama_token new_token_id;
    constexpr int max_reply_tokens    = 1024;
    constexpr int extra_cutoff_tokens = 32;
    int generated_token_count         = 0;

    // Loop until end-of-generation token
    // 終了トークンに達するまでループ
    while (true) {
        // Decode the batch
        // バッチをデコード
        if (llama_decode(m_ctx, batch)) {
            Q_EMIT generationError(QStringLiteral("failed to decode"));
            break;
        }
        // Sample next token
        // 次のトークンをサンプリング
        new_token_id = llama_sampler_sample(m_sampler, m_ctx, -1);
        // Stop if end-of-generation token
        // 終了トークンなら終了
        if (llama_token_is_eog(m_model, new_token_id)) {
            break;
        }

        // Convert token to text
        // トークンをテキストに変換
        char buf[256] = {};
        const int n = llama_token_to_piece(m_model, new_token_id, buf,
                                           sizeof(buf), 0, true);
        if (n < 0) {
            Q_EMIT generationError(QStringLiteral("failed to convert token to piece"));
            break;
        }

        std::string piece(buf, n);
        response += piece;

        // Emit partial progress
        // 部分的な進捗をシグナルで通知
        Q_EMIT partialResponseReady(QString::fromStdString(response));

        // Prepare next batch
        // 次のバッチを作成
        batch = llama_batch_get_one(&new_token_id, 1);

        // Cutoff if response is too long
        // レスポンスが長すぎる場合は打ち切り
        ++generated_token_count;
        if (generated_token_count > max_reply_tokens) {
            if (piece.find('\n') != std::string::npos) {
                qDebug() << "Cut off at newline";
                break;
            } else if (generated_token_count > max_reply_tokens + extra_cutoff_tokens) {
                qDebug() << "Cut off after extra tokens";
                break;
            }
        }
    }

    // Emit final response
    // 最終的なレスポンスを通知
    Q_EMIT generationFinished(QString::fromStdString(response));
}

// Initialize sampler with default parameters
// デフォルトパラメータでサンプラーを初期化
void LlamaResponseGenerator::initializeSampler()
{
    m_sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(m_sampler, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(m_sampler, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(m_sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
}
