#include "llamaresponsegenerator.h"
#include <QCoreApplication>

int main(int argc, char *argv[])
{
    QCoreApplication app(argc, argv);

    qSetMessagePattern("[%{file}:%{line}] %{message}");

    LlamaResponseGenerator llamaResponseGenerator;

    QRemoteObjectHost srcNode(QUrl(QStringLiteral("tcp://0.0.0.0:12345")));
    srcNode.enableRemoting(&llamaResponseGenerator);

    return app.exec();
}
