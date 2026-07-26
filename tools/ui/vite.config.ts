import tailwindcss from '@tailwindcss/vite';
import { sveltekit } from '@sveltejs/kit/vite';
import { SvelteKitPWA } from '@vite-pwa/sveltekit';
import { dirname, resolve } from 'path';
import { fileURLToPath } from 'url';

import { defineConfig, loadEnv, searchForWorkspaceRoot } from 'vite';
import { storybookTest } from '@storybook/addon-vitest/vitest-plugin';
import { splashScreenPlugin } from './scripts/vite-plugin-splash-screen';
import { buildInfoPlugin } from './scripts/vite-plugin-build-info';
import { relativizeBasePlugin } from './scripts/vite-plugin-relativize-base';
import { nerdamerPlugin } from './scripts/vite-plugin-nerdamer';
import { playwright } from '@vitest/browser-playwright';
import { SVELTEKIT_PWA_OPTIONS } from './src/lib/constants/pwa';

const __dirname = dirname(fileURLToPath(import.meta.url));

const createServerProxy = (mode: string) => {
	const serverOrigin =
		loadEnv(mode, process.cwd(), 'VITE_').VITE_PUBLIC_SERVER_ORIGIN || 'http://localhost:8080';

	return {
		'/v1': serverOrigin,
		'/props': serverOrigin,
		'/models': serverOrigin,
		'/tools': serverOrigin,
		'/slots': serverOrigin,
		'/cors-proxy': serverOrigin
	};
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const browserBaseConfig: any = {
	enabled: true,
	provider: playwright({
		launchOptions: {
			args: ['--no-sandbox']
		}
	}),
	instances: [{ browser: 'chromium' }]
};

export default defineConfig(({ mode }) => ({
	resolve: {
		alias: {
			'katex-fonts': resolve('node_modules/katex/dist/fonts')
		}
	},

	build: {
		assetsInlineLimit: 32000,
		chunkSizeWarningLimit: 3072,
		minify: true
	},

	plugins: [
		tailwindcss(),
		sveltekit(),
		SvelteKitPWA(SVELTEKIT_PWA_OPTIONS),
		splashScreenPlugin(),
		buildInfoPlugin(),
		nerdamerPlugin(),
		relativizeBasePlugin()
	],

	test: {
		projects: [
			{
				extends: './vite.config.ts',
				test: {
					name: 'client',
					browser: browserBaseConfig,
					include: ['tests/client/**/*.svelte.{test,spec}.{js,ts}'],
					setupFiles: ['./vitest-setup-client.ts']
				}
			},

			{
				extends: './vite.config.ts',
				test: {
					name: 'unit',
					environment: 'node',
					include: ['tests/unit/**/*.{test,spec}.{js,ts}']
				}
			},

			{
				extends: './vite.config.ts',
				test: {
					name: 'ui',
					browser: { ...browserBaseConfig, instances: [{ browser: 'chromium', headless: true }] },
					setupFiles: ['./.storybook/vitest.setup.ts']
				},
				plugins: [
					storybookTest({
						storybookScript: 'pnpm run storybook --no-open'
					})
				]
			}
		]
	},

	server: {
		proxy: createServerProxy(mode),
		headers: {
			'Cross-Origin-Embedder-Policy': 'require-corp',
			'Cross-Origin-Opener-Policy': 'same-origin'
		},
		fs: {
			allow: [searchForWorkspaceRoot(process.cwd()), resolve(__dirname, 'tests')]
		}
	}
}));
