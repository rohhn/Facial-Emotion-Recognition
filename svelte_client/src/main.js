import App from './App.svelte';

const app = new App({
	target: document.getElementById('main'),
	props: {
		name: 'fer_app'
	}
});

export default app;