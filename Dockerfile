FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src
COPY Directory.Build.props ./
COPY src/KernelRag.Core/KernelRag.Core.csproj src/KernelRag.Core/
COPY src/KernelRag.Api/KernelRag.Api.csproj src/KernelRag.Api/
RUN dotnet restore src/KernelRag.Api/KernelRag.Api.csproj
COPY src/KernelRag.Core/ src/KernelRag.Core/
COPY src/KernelRag.Api/ src/KernelRag.Api/
RUN dotnet publish src/KernelRag.Api/KernelRag.Api.csproj -c Release -o /app --no-restore

FROM mcr.microsoft.com/dotnet/aspnet:8.0
WORKDIR /app
COPY --from=build /app .
ENV ASPNETCORE_URLS=http://+:8000
EXPOSE 8000
USER app
ENTRYPOINT ["dotnet", "KernelRag.Api.dll"]
