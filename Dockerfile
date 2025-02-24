# 使用官方的Python基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 复制依赖文件到工作目录
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码到工作目录
COPY . .

# 暴露端口
EXPOSE 8000

# 运行应用
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
