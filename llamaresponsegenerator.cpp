#include "llamaresponsegenerator.h"
#include <QDebug>

// コンストラクタ
LlamaResponseGenerator::LlamaResponseGenerator(QObject *parent,
                                               llama_model *model,
                                               llama_context *ctx)
    : LlamaResponseGeneratorSimpleSource(parent) // 親クラス(生成コード)のコンストラクタ呼び出し
    , m_model(model)
    , m_ctx(ctx)
{
}

// デストラクタ
LlamaResponseGenerator::~LlamaResponseGenerator()
{
    // sampler が生成されていたら解放
    if (m_sampler) {
        llama_sampler_free(m_sampler);
        m_sampler = nullptr;
    }
}

// repファイル由来の generate(...) スロット
void LlamaResponseGenerator::generate(const QString &request)
{
    // 最初の生成時にサンプラー初期化
    if (!m_sampler) {
        initializeSampler();
    }

    // ここから先は、元の generate() 実装とほぼ同じ
    std::string response;
    std::string promptStd = request.toStdString();

    // --- (以下、トークナイズとデコードループの処理) ---
    const int n_prompt_tokens = -llama_tokenize(
        m_model,
        promptStd.c_str(),
        promptStd.size(),
        nullptr,
        0,
        true,
        true
        );

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

    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    llama_token new_token_id;
    constexpr int max_reply_tokens    = 1024;
    constexpr int extra_cutoff_tokens = 32;
    int generated_token_count         = 0;

    while (true) {
        if (llama_decode(m_ctx, batch)) {
            Q_EMIT generationError(QStringLiteral("failed to decode"));
            break;
        }
        new_token_id = llama_sampler_sample(m_sampler, m_ctx, -1);
        if (llama_token_is_eog(m_model, new_token_id)) {
            break;
        }

        char buf[256] = {};
        const int n = llama_token_to_piece(m_model, new_token_id, buf,
                                           sizeof(buf), 0, true);
        if (n < 0) {
            Q_EMIT generationError(QStringLiteral("failed to convert token to piece"));
            break;
        }

        std::string piece(buf, n);
        response += piece;

        // 部分的な進捗をシグナルで通知
        Q_EMIT partialResponseReady(QString::fromStdString(response));

        // 次のトークンをバッチに
        batch = llama_batch_get_one(&new_token_id, 1);

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

    // 生成完了シグナル
    Q_EMIT generationFinished(QString::fromStdString(response));
}

// サンプラー初期化処理
void LlamaResponseGenerator::initializeSampler()
{
    m_sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(m_sampler, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(m_sampler, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(m_sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
}
