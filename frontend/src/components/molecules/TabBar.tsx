import { BorderBox } from "../atoms/BorderBox";

interface Props {
  value: "all" | "waiting" | "running" | "terminated";
}

export function TabBar() {
  return (
    <BorderBox className="flex h-8 w-72 flex-row items-center justify-evenly divide-x text-xs">
      <label>All</label>
      <label>Waiting</label>
      <label>Running</label>
      <label>Terminated</label>
    </BorderBox>
  );
}
