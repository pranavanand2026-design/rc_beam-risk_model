import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",
  reactStrictMode: true,
  experimental: {
    optimizePackageImports: [
      "lucide-react",
      "@tanstack/react-query",
    ],
  },
};

export default nextConfig;
